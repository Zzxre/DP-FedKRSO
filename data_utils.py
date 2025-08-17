import numpy as np
from collections import defaultdict
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer
)


def load_and_preprocess_data(task, args):

    if "mnli" in task:
        dataset = load_dataset("glue", "mnli")
    else:
        dataset = load_dataset("glue", task)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def tokenize_function(examples):

        # Handle different input formats
        if "premise" in examples and "hypothesis" in examples:
            # MNLI and similar tasks
            return tokenizer(
                examples["premise"],
                examples["hypothesis"],
                truncation=True,
                padding="max_length",
                max_length=args.max_seq_length,
            )
        elif "question" in examples and "sentence" in examples:
            # QNLI and similar tasks
            return tokenizer(
                examples["question"],
                examples["sentence"],
                truncation=True,
                padding="max_length",
                max_length=args.max_seq_length,
            )
        elif "sentence1" in examples and "sentence2" in examples:
            # MRPC, STS-B
            return tokenizer(
                examples["sentence1"],
                examples["sentence2"],
                truncation=True,
                padding="max_length",
                max_length=args.max_seq_length,
            )
        elif "question1" in examples and "question2" in examples:
            # QQP
            return tokenizer(
                examples["question1"],
                examples["question2"],
                truncation=True,
                padding="max_length",
                max_length=args.max_seq_length,
            )
        elif "sentence" in examples:
            # CoLA, SST-2
            return tokenizer(
                examples["sentence"],
                truncation=True,
                padding="max_length",
                max_length=args.max_seq_length,
            )
        else:
            raise ValueError(f"Unexpected format for task {task}")

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    if task == "cola":
        tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])
    elif task == "sst2":
        tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])
    elif task == "mrpc":
        tokenized_datasets = tokenized_datasets.remove_columns(
            ["sentence1", "sentence2", "idx"]
        )
    elif task == "qqp":
        tokenized_datasets = tokenized_datasets.remove_columns(
            ["question1", "question2", "idx"]
        )
    elif task == "stsb":
        tokenized_datasets = tokenized_datasets.remove_columns(
            ["sentence1", "sentence2", "idx"]
        )
    elif task == "qnli":
        tokenized_datasets = tokenized_datasets.remove_columns(
            ["question", "sentence", "idx"]
        )
    elif task == "rte":
        tokenized_datasets = tokenized_datasets.remove_columns(
            ["sentence1", "sentence2", "idx"]
        )
    elif task == "wnli":
        tokenized_datasets = tokenized_datasets.remove_columns(
            ["sentence1", "sentence2", "idx"]
        )
    elif task == "mnli_matched" or task == "mnli_mismatched" or task == "mnli":
        tokenized_datasets = tokenized_datasets.remove_columns(
            ["premise", "hypothesis", "idx"]
        )
    else:
        raise ValueError(f"Unexpected task {task}")

    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    if (
        task == "cola"
        or task == "sst2"
        or task == "mrpc"
        or task == "qqp"
        or task == "stsb"
        or task == "qnli"
        or task == "rte"
        or task == "wnli"
    ):
        train_dataset = tokenized_datasets["train"]
        val_dataset = tokenized_datasets["validation"]
        test_dataset = tokenized_datasets["test"]
    elif task == "mnli_matched":
        train_dataset = tokenized_datasets["train"]
        val_dataset = tokenized_datasets["validation_matched"]
        test_dataset = tokenized_datasets["test_matched"]
    elif task == "mnli_mismatched":
        train_dataset = tokenized_datasets["train"]
        val_dataset = tokenized_datasets["validation_mismatched"]
        test_dataset = tokenized_datasets["test_mismatched"]

    return train_dataset, val_dataset, test_dataset


def create_dataloader(dataset, args):
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False)


def create_client_dataloaders_nlg(dataset, args):
    client_data = [[] for _ in range(args.num_clients)]
    for data in dataset:
        client_idx = np.random.randint(args.num_clients)
        client_data[client_idx].append(data)
    return client_data


def dirichlet_partition(labels, n_clients=10, alpha=0.5, seed=42):
    """
    使用 Dirichlet 分布将数据划分为多个子集（模拟 non-IID，局部种子安全）

    参数：
        labels: numpy 数组，一维标签数组
        n_clients: 客户端数量
        alpha: Dirichlet 分布的浓度参数（越小越 non-IID）
        seed: 本函数内部使用的局部种子，不影响外部随机状态

    返回：
        dict: {client_id: [sample_indices]}
    """
    rng = np.random.default_rng(seed)  # 独立随机生成器，不影响全局

    n_classes = len(np.unique(labels))
    label_indices = [np.where(labels == y)[0] for y in range(n_classes)]
    client_data_indices = defaultdict(list)

    for c, indices in enumerate(label_indices):
        rng.shuffle(indices)  # 用局部 RNG 随机打乱
        proportions = rng.dirichlet(alpha=np.ones(n_clients) * alpha)
        proportions = np.clip(proportions, 1e-5, None)
        proportions = proportions / proportions.sum()
        split_counts = (proportions * len(indices)).astype(int)

        while split_counts.sum() < len(indices):
            split_counts[rng.integers(0, n_clients)] += 1
        while split_counts.sum() > len(indices):
            split_counts[rng.integers(0, n_clients)] -= 1

        splits = np.split(indices, np.cumsum(split_counts)[:-1])

        for client_id, split in enumerate(splits):
            client_data_indices[client_id].extend(split.tolist())

    return client_data_indices


def create_noniid_client_dataloaders(dataset, args, labels):
    client_data = [[] for _ in range(args.num_clients)]
    index = dirichlet_partition(labels=labels, n_clients=args.num_clients, alpha=args.alpha)
    for client_id in index:
        for data_idx in index[client_id]:
            client_data[client_id].append(dataset[data_idx])
    return [
        DataLoader(cd, batch_size=args.batch_size, shuffle=True) if len(cd)!=0 else None for cd in client_data
    ]


def create_client_dataloaders(dataset, args):
    client_data = [[] for _ in range(args.num_clients)]
    for data in dataset:
        client_idx = np.random.randint(args.num_clients)
        client_data[client_idx].append(data)
    return [
        DataLoader(cd, batch_size=args.batch_size, shuffle=True) for cd in client_data
    ]