from peft import (
    get_peft_model,
    TaskType,
    LoraConfig,
)
import torch


def create_peft_model(model, args):
    if args.lora_r == 0:
        return model
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_rslora=args.rslora,
        # target_modules=["query", "value", "key", ],
        target_modules=["query", "value", "key", "intermediate.dense", "output.dense"],
    )

    model = get_peft_model(model, peft_config)
    return model


def create_peft_FFA_model(model, args):
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_rslora=args.rslora,
        target_modules=["query", "value", "key", "intermediate.dense", "output.dense"],
    )
    model = get_peft_model(model, peft_config)

    # Make LoRA A matrices non-trainable
    for name, param in model.named_parameters():
        if "lora_A" in name:
            param.requires_grad = False

    return model


def create_RSO_model(model, args):
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_rslora=args.rslora,
        target_modules=["query", "value", "key",
                        "intermediate.dense", "output.dense"
                        ],
    )
    model = get_peft_model(model, peft_config)

    def orthonormal_gaussian_init(tensor, shape):
        """ Initialize LoRA-A with a Gaussian matrix ensuring E[A^T A] = I """
        torch.nn.init.normal_(tensor, mean=0, std=1.0 / shape ** 0.5, generator=g)

    import random
    g = torch.Generator(device=args.device)
    if hasattr(args, 'kseed') and args.kseed != -1:
        g.manual_seed(random.choice(args.seed_list))

    # Make LoRA A matrices non-trainable
    for name, param in model.named_parameters():
        if "lora_A" in name:
            param.requires_grad = False
            orthonormal_gaussian_init(param, param.shape[0])
    return model
