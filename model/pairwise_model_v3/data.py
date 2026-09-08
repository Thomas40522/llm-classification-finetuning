import json
import torch
from tqdm.auto import tqdm
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

def build_reward_input(row, response_key):
    conversation = []

    for prompt, response_a, response_b in zip(
        row["prompt_parsed"],
        row["response_a_parsed"],
        row["response_b_parsed"]
    ):
        if prompt is not None:
            conversation.append(f"User: {prompt}")

        response = response_a if response_key == "a" else response_b

        if response is not None:
            conversation.append(f"Assistant: {response}")

    return "\n".join(conversation)


def tokenize_head_tail(
    text,
    tokenizer,
    max_length=512,
    head_ratio=0.5
):
    # Tokenize without special tokens
    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False
    )

    # Reserve space for [CLS] and [SEP]
    available_length = max_length - 2

    # Short sequence: keep everything
    if len(token_ids) <= available_length:

        input_ids = (
            [tokenizer.cls_token_id]
            + token_ids
            + [tokenizer.sep_token_id]
        )

    # Long sequence: keep head + tail
    else:

        head_length = int(
            available_length * head_ratio
        )

        tail_length = (
            available_length - head_length
        )

        head = token_ids[:head_length]
        tail = token_ids[-tail_length:]

        input_ids = (
            [tokenizer.cls_token_id]
            + head
            + tail
            + [tokenizer.sep_token_id]
        )

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids)
    }


def tokenize_dataframe(df, tokenizer, max_length=512):
    enc_a = []
    enc_b = []

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Tokenizing"
    ):
        enc_a.append(
            tokenize_head_tail(
                row["input_a"],
                tokenizer,
                max_length
            )
        )

        enc_b.append(
            tokenize_head_tail(
                row["input_b"],
                tokenizer,
                max_length
            )
        )

    labels = df["label"].tolist()

    return enc_a, enc_b, labels

def tokenize_dataframe_test(df, tokenizer, max_length=512):
    enc_a = []
    enc_b = []

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Tokenizing"
    ):
        enc_a.append(
            tokenize_head_tail(
                row["input_a"],
                tokenizer,
                max_length
            )
        )

        enc_b.append(
            tokenize_head_tail(
                row["input_b"],
                tokenizer,
                max_length
            )
        )

    return enc_a, enc_b


class PreferenceDataset(Dataset):

    def __init__(self, enc_a, enc_b, labels):
        self.enc_a = enc_a
        self.enc_b = enc_b
        self.labels = torch.tensor(
            labels,
            dtype=torch.long
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.enc_a[idx],
            self.enc_b[idx],
            self.labels[idx]
        )

class PreferenceDatasetTest(Dataset):

    def __init__(self, enc_a, enc_b):
        self.enc_a = enc_a
        self.enc_b = enc_b


    def __len__(self):
        return len(self.enc_a)

    def __getitem__(self, idx):
        return (
            self.enc_a[idx],
            self.enc_b[idx]
        )


def collate_fn(batch, tokenizer):
    batch_a = [item[0] for item in batch]
    batch_b = [item[1] for item in batch]

    labels = torch.stack(
        [item[2] for item in batch]
    )

    enc_a = tokenizer.pad(
        batch_a,
        padding=True,
        return_tensors="pt"
    )

    enc_b = tokenizer.pad(
        batch_b,
        padding=True,
        return_tensors="pt"
    )

    enc_a.pop("token_type_ids", None)
    enc_b.pop("token_type_ids", None)

    return enc_a, enc_b, labels

def collate_fn_test(batch, tokenizer):
    batch_a = [item[0] for item in batch]
    batch_b = [item[1] for item in batch]

    enc_a = tokenizer.pad(
        batch_a,
        padding=True,
        return_tensors="pt"
    )

    enc_b = tokenizer.pad(
        batch_b,
        padding=True,
        return_tensors="pt"
    )

    enc_a.pop("token_type_ids", None)
    enc_b.pop("token_type_ids", None)

    return enc_a, enc_b

def preprocess_dataframe(df, test_size, is_full_dataset, dataset_limit, random_state):
    # =====================================
    # Load and process data
    # =====================================

    df["prompt_parsed"] = df["prompt"].apply(json.loads)
    df["response_a_parsed"] = df["response_a"].apply(json.loads)
    df["response_b_parsed"] = df["response_b"].apply(json.loads)

    target_cols = [
        "winner_model_a",
        "winner_model_b",
        "winner_tie"
    ]

    df["label"] = df[target_cols].values.argmax(axis=1)

    if not is_full_dataset:
    # Keep the same 10,000-example experiment from the notebook.
        df_part = df.iloc[:dataset_limit].copy()

        train_df, val_df = train_test_split(
            df_part,
            test_size=test_size,
            random_state=random_state,
            stratify=df_part["label"]
        )
    else:
        train_df, val_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df["label"]
        )


    train_df["input_a"] = train_df.apply(
        lambda row: build_reward_input(row, "a"),
        axis=1
    )

    train_df["input_b"] = train_df.apply(
        lambda row: build_reward_input(row, "b"),
        axis=1
    )

    val_df["input_a"] = val_df.apply(
        lambda row: build_reward_input(row, "a"),
        axis=1
    )

    val_df["input_b"] = val_df.apply(
        lambda row: build_reward_input(row, "b"),
        axis=1
    )

    return train_df, val_df

def preprocess_dataframe_test(df):
    # =====================================
    # Load and process data
    # =====================================

    df["prompt_parsed"] = df["prompt"].apply(json.loads)
    df["response_a_parsed"] = df["response_a"].apply(json.loads)
    df["response_b_parsed"] = df["response_b"].apply(json.loads)


    df["input_a"] = df.apply(
        lambda row: build_reward_input(row, "a"),
        axis=1
    )

    df["input_b"] = df.apply(
        lambda row: build_reward_input(row, "b"),
        axis=1
    )

    return df

def load_dataframe_train(train_df, tokenizer, max_length, batch_size):
    train_dataset = PreferenceDataset(
        *tokenize_dataframe(
            train_df,
            tokenizer,
            max_length
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )

    return train_loader


def load_dataframe_val(val_df, tokenizer, max_length, batch_size):
    val_dataset = PreferenceDataset(
        *tokenize_dataframe(
            val_df,
            tokenizer,
            max_length
        )
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )

    return val_loader

def load_dataframe_test(test_df, tokenizer, max_length, batch_size):
    test_dataset = PreferenceDatasetTest(
        *tokenize_dataframe_test(
            test_df,
            tokenizer,
            max_length
        )
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn_test(batch, tokenizer)
    )

    return test_loader