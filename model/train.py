import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from data import (
    build_reward_input,
    tokenize_dataframe,
    PreferenceDataset,
    collate_fn,
)
from model import RewardModel, preference_probabilities


# =====================================
# Configuration
# =====================================

TRAIN_PATH = "../data/train.csv"
MODEL_NAME = "microsoft/deberta-v3-small"

MAX_LENGTH = 512
BATCH_SIZE = 2
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5

CHECKPOINT_DIR = "../models/reward_model"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# =====================================
# Load and process data
# =====================================

df = pd.read_csv(TRAIN_PATH)

df["prompt_parsed"] = df["prompt"].apply(json.loads)
df["response_a_parsed"] = df["response_a"].apply(json.loads)
df["response_b_parsed"] = df["response_b"].apply(json.loads)

target_cols = [
    "winner_model_a",
    "winner_model_b",
    "winner_tie"
]

df["label"] = df[target_cols].values.argmax(axis=1)

# Keep the same 10,000-example experiment from the notebook.
# df_small = df.iloc[:10000].copy()

# train_df, val_df = train_test_split(
#     df,
#     test_size=0.01,
#     random_state=42,
#     stratify=df["label"]
# )

# train all the data, no validation split for now
train_df = df

train_df["input_a"] = train_df.apply(
    lambda row: build_reward_input(row, "a"),
    axis=1
)

train_df["input_b"] = train_df.apply(
    lambda row: build_reward_input(row, "b"),
    axis=1
)

# val_df["input_a"] = val_df.apply(
#     lambda row: build_reward_input(row, "a"),
#     axis=1
# )

# val_df["input_b"] = val_df.apply(
#     lambda row: build_reward_input(row, "b"),
#     axis=1
# )


# =====================================
# Model and tokenizer
# =====================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

backbone = AutoModel.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

reward_model = RewardModel(backbone).to(device)

optimizer = torch.optim.AdamW(
    reward_model.parameters(),
    lr=LEARNING_RATE
)


# =====================================
# Tokenize ONCE
# =====================================

train_dataset = PreferenceDataset(
    *tokenize_dataframe(
        train_df,
        tokenizer,
        MAX_LENGTH
    )
)

# val_dataset = PreferenceDataset(
#     *tokenize_dataframe(
#         val_df,
#         tokenizer,
#         MAX_LENGTH
#     )
# )

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=lambda batch: collate_fn(batch, tokenizer)
)

# val_loader = DataLoader(
#     val_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=False,
#     collate_fn=lambda batch: collate_fn(batch, tokenizer)
# )


# =====================================
# Training
# =====================================

train_losses = []
train_steps = []

for epoch in range(NUM_EPOCHS):

    reward_model.train()

    running_loss = 0.0
    num_examples = 0

    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"
    )

    for enc_a, enc_b, labels in progress_bar:

        enc_a = {
            k: v.to(device)
            for k, v in enc_a.items()
        }

        enc_b = {
            k: v.to(device)
            for k, v in enc_b.items()
        }

        labels = labels.to(device)

        optimizer.zero_grad()

        reward_a = reward_model(**enc_a)
        reward_b = reward_model(**enc_b)

        tie_param = reward_model.get_tie_param()

        prob_a, prob_b, prob_tie = preference_probabilities(
            reward_a,
            reward_b,
            tie_param
        )

        probabilities = torch.stack(
            [prob_a, prob_b, prob_tie],
            dim=1
        )

        true_probabilities = probabilities[
            torch.arange(
                labels.size(0),
                device=labels.device
            ),
            labels
        ]

        loss = -torch.log(true_probabilities).mean()

        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        train_steps.append(len(train_losses))

        batch_size = labels.size(0)

        running_loss += loss.item() * batch_size
        num_examples += batch_size

        average_loss = running_loss / num_examples

        progress_bar.set_postfix(
            loss=f"{average_loss:.4f}"
        )

    train_loss = running_loss / num_examples

    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS} "
        f"- Train Loss: {train_loss:.4f}"
    )

    checkpoint_path = (
        f"{CHECKPOINT_DIR}/epoch_{epoch + 1}.pt"
    )

    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": reward_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
        },
        checkpoint_path
    )

    print(f"Checkpoint saved to: {checkpoint_path}")