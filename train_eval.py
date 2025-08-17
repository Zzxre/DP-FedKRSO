import torch
import wandb
import math
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score
from torch.cuda.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from optimizer import AdamW


torch.set_printoptions(precision=4)


def train_client(model, dataloader, args):
    device = args.device
    model.to(device)

    if args.agg_type == 'rso':
        optimizer = AdamW(model.named_parameters(),
                          lr=args.lr,
                          weight_decay=args.weight_decay,
                          mode='efficient',
                          scaling_factor=(args.lora_alpha / args.lora_r),
                          interval=args.interval,
                          args=args)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    ###################
    if args.dp:
        print("*****************DP is enabled.*****************")
        # enter PrivacyEngine
        model.train()
        privacy_engine = PrivacyEngine()
        model, optimizer, dataloader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=dataloader,
            noise_multiplier=args.noise_multiplier,
            max_grad_norm=args.clip_threshold,
        )
    ###################
    total_steps = args.local_steps

    num_warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps
    )

    total_loss = 0

    scaler = GradScaler()
    model.train()
    step = 0
    while step < total_steps:
        if args.dp:
            args.amp = False
            # Use BatchMemoryManager to handle batch memory management for DP
            with BatchMemoryManager(
                data_loader=dataloader,
                max_physical_batch_size=args.physical_bs,
                optimizer=optimizer,
            ) as new_dataloader:
                for data in new_dataloader:
                    if args.amp:
                        with autocast():
                            loss = model(**{k: v.to(device) for k, v in data.items()}).loss
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss = model(**{k: v.to(device) for k, v in data.items()}).loss
                        loss.backward()
                        optimizer.step()

                    scheduler.step()
                    optimizer.zero_grad()
                    step += 1
                    total_loss += loss.item()
                    if step >= args.local_steps:
                        break
        else:
            for data in dataloader:
                if args.amp:
                    with autocast():
                        loss = model(**{k: v.to(device) for k, v in data.items()}).loss
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss = model(**{k: v.to(device) for k, v in data.items()}).loss
                    loss.backward()
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                step += 1
                total_loss += loss.item()
                if step >= args.local_steps:
                    break

    return model, total_loss/total_steps


def calculate_metrics(all_true_labels, all_predictions, task):
    if task == "cola":
        return accuracy_score(all_true_labels, all_predictions), matthews_corrcoef(
            all_true_labels, all_predictions
        )
    elif task in ["sst2", "qnli", "rte", "wnli", "mnli_matched", "mnli_mismatched"]:
        return accuracy_score(all_true_labels, all_predictions), None
    elif task in ["mrpc", "qqp"]:
        return f1_score(all_true_labels, all_predictions), accuracy_score(
            all_true_labels, all_predictions
        )
    elif task == "stsb":
        return (
            pearsonr(all_true_labels, all_predictions)[0],
            spearmanr(all_true_labels, all_predictions)[0],
        )
    else:
        raise ValueError(f"Unknown task: {task}")


def evaluate_global_model(global_model, dataloader, args, r, max_metric1, max_metric2):
    device = args.device
    # global_model.to(device)

    global_model.eval()
    eval_loss = 0
    all_predictions = []
    all_true_labels = []

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():

            outputs = global_model(**batch)

            eval_loss += outputs.loss.detach().cpu().numpy()

            if args.task == "stsb":
                predictions = outputs.logits.squeeze().cpu().numpy()
            else:
                predictions = outputs.logits.argmax(dim=-1).cpu().numpy()
            all_predictions.extend(predictions)
            all_true_labels.extend(batch["labels"].cpu().numpy())

    eval_loss /= len(dataloader)

    # Calculate the metrics for the specific task
    metric1, metric2 = calculate_metrics(all_true_labels, all_predictions, args.task)

    if metric1 > max_metric1:
        max_metric1 = metric1

    if metric2 is not None and metric2 > max_metric2:
        max_metric2 = metric2

    print(f"{args.task} - Eval Loss: {eval_loss:.4f}, Metric 1: {metric1:.4f}")
    if metric2 is not None:
        print(f"{args.task} - Metric 2: {metric2:.4f}")
    print(f"{args.task} - Max Metric 1: {max_metric1:.4f}")
    if max_metric2 is not None:
        print(f"{args.task} - Max Metric 2: {max_metric2:.4f}")

    if wandb.run:
        wandb.log(
            {
                "round": r,
                f"eval_loss": eval_loss,
                f"metric1": metric1,
                f"metric2": metric2 if metric2 is not None else 0,
                f"max_metric1": max_metric1,
                f"max_metric2": max_metric2 if max_metric2 is not None else 0,
            }
        )

    return max_metric1, max_metric2

