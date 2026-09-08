import json
import torch
from tqdm.auto import tqdm
from torch.utils.data import Dataset

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