import random
from copy import deepcopy
from typing import Union, Tuple

import torch
from torch.optim import Optimizer


class SGD(Optimizer):
    def __init__(
            self,
            named_params,
            lr: Union[float, torch.Tensor] = 1e-3,
            scaling_factor: float = 2.,
            weight_decay: float = 0,
            momentum=0.,
            mode='efficient',
            interval=100,
            optimizer_states="reset",
            norm_limit=True,
            args=None,
    ):
        names = []
        params = []
        for n, p in named_params:
            names.append(n)
            params.append(p)
        self.mode = mode
        self.interval = interval
        self.optimizer_states = optimizer_states
        self.norm_limit = norm_limit
        self.args = args

        defaults = dict(
            lr=lr,
            names=names,
            scaling_factor=scaling_factor,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        g = torch.Generator(self.args.device)
        if hasattr(self.args, 'kseed') and self.args.kseed != -1:
            g.manual_seed(random.choice(self.args.seed_list))
        for group in self.param_groups:
            scaling_factor = group["scaling_factor"]
            momentum = group['momentum']
            param_list = []
            param_dict = dict(zip(group["names"], group["params"]))
            for n, p in param_dict.items():

                if 'lora' in n:
                    param_list.append(p)
                    if len(param_list) == 2:
                        base_name = n[: n.find('lora')]
                        name = base_name + "base_layer.weight"
                        if self.mode == 'full':
                            size = param_dict[name].shape
                        else:
                            size = param_dict[n].shape
                    else:
                        continue
                elif p.grad is None:
                    continue
                else:
                    name = n
                    size = p.shape

                state = self.state[name]
                if len(state) == 0:
                    state['step'] = torch.tensor(0.0)
                    if momentum != 0:
                        state["momentum"] = torch.zeros(size).to(p.device).to(p.dtype)

                state['step'] += 1

                # step
                if len(param_list) == 2:
                    if self.mode == 'full':
                        grad = (param_list[1].grad @ param_list[0]) / scaling_factor
                    else:
                        grad = param_list[1].grad / scaling_factor
                else:
                    grad = p.grad

                if momentum != 0:
                    buf = state["momentum"]
                    buf.mul_(momentum).add_(grad, alpha=1)
                    grad = buf

                step_size = group['lr']

                if len(param_list) != 2:
                    p.mul_(1 - group["weight_decay"] * group["lr"])
                    p.add_(grad, alpha=-step_size)
                else:
                    param_dict[name].mul_(1 - group["weight_decay"] * group["lr"])
                    if self.mode == 'full':
                        param_dict[name].add_(
                            grad,
                            alpha=-step_size
                        )
                    else:
                        grad = grad @ param_list[0]

                        param_dict[name].add_(
                            grad,
                            alpha=-step_size
                        )
                    if state['step'] % self.interval == 0:
                        old_param = deepcopy(param_list[0])
                        torch.nn.init.normal_(param_list[0], mean=0, std=1.0 / param_list[0].shape[0] ** 0.5,
                                              generator=g)
                        if self.mode != 'full' and 'momentum' in state:
                            if self.optimizer_states == 'reset':
                                torch.nn.init.zeros_(state['momentum'])
                            elif self.optimizer_states == 'transform':
                                state['momentum'] = state['momentum'] @ old_param @ param_list[0].T * (
                                            param_list[0].shape[0] / param_list[0].shape[1])
                    param_list = []
        return loss


class AdamW(Optimizer):
    def __init__(
            self,
            named_params,
            lr: Union[float, torch.Tensor] = 1e-3,
            scaling_factor: float = 2.,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 0,
            mode='efficient',
            interval=100,
            optimizer_states="reset",
            args=None
    ):
        names = []
        params = []
        for n, p in named_params:
            names.append(n)
            params.append(p)
        self.mode = mode
        self.interval = interval
        self.optimizer_states = optimizer_states
        self.args = args
        defaults = dict(
            lr=lr,
            names=names,
            scaling_factor=scaling_factor,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            scaling_factor = group["scaling_factor"]

            g = torch.Generator(self.args.device)
            if hasattr(self.args, 'kseed') and self.args.kseed != -1:
                g.manual_seed(random.choice(self.args.seed_list))
            param_list = []
            param_dict = dict(zip(group["names"], group["params"]))
            for n, p in param_dict.items():
                if 'lora' in n:
                    if 'lora_B' in n:
                        size = param_dict[n].shape
                    param_list.append(p)
                    if len(param_list) == 2:
                        base_name = n[: n.find('lora')]
                        name = base_name + "base_layer.weight"
                        if self.mode == 'full':
                            size = param_dict[name].shape
                    else:
                        continue
                elif p.grad is None:
                    continue
                else:
                    name = n
                    size = p.shape

                state = self.state[name]
                # Lazy state initialization
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0)

                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros(size).to(p.device).to(p.dtype)

                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros(size).to(p.device).to(p.dtype)

                if len(param_list) == 2:
                    if self.mode == 'full':
                        grad = (param_list[1].grad @ param_list[0]) / scaling_factor
                    else:
                        grad = param_list[1].grad / scaling_factor
                else:
                    grad = p.grad

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.lerp_(grad, 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = group['lr']
                denom = (exp_avg_sq.sqrt() / bias_correction2 ** 0.5).add_(group['eps'])

                if len(param_list) != 2:
                    p.mul_(1 - group["weight_decay"] * group["lr"])
                    p.addcdiv_(exp_avg / bias_correction1, denom, value=-step_size)
                else:
                    param_dict[name].mul_(1 - group["weight_decay"] * group["lr"])
                    if self.mode == 'full':
                        param_dict[name].add_(
                            (exp_avg / bias_correction1 / denom),
                            alpha=-step_size
                        )
                    else:
                        param_dict[name].add_(
                            (exp_avg / bias_correction1 / denom).to(param_list[0].dtype) @ param_list[0],
                            alpha=-step_size
                        )
                    if state['step'] % self.interval == 0:
                        old_param = deepcopy(param_list[0])
                        torch.nn.init.normal_(param_list[0], mean=0, std=1.0 / param_list[0].shape[0] ** 0.5,
                                              generator=g)
                        if self.mode != 'full' and 'exp_avg' in state:
                            if self.optimizer_states == 'reset':
                                torch.nn.init.zeros_(state['exp_avg'])
                                torch.nn.init.zeros_(state['exp_avg_sq'])
                            elif self.optimizer_states == 'transform':
                                state['exp_avg'] = state['exp_avg'] @ old_param @ param_list[0].T * (
                                            param_list[0].shape[0] / param_list[0].shape[1])
                                # state['exp_avg_sq'] = state['exp_avg_sq'] @ old_param @ param_list[0].T
                    param_list = []

        return loss
