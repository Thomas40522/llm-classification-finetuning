import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from data import (
    preprocess_dataframe,
    tokenize_dataframe_train
)
from model import PairwisePreferenceModel


# =====================================
# Configuration
# =====================================

TRAIN_PATH = "../../data/train.csv"
MODEL_NAME = "microsoft/deberta-v3-small"

MAX_LENGTH = 512
BATCH_SIZE = 2
NUM_EPOCHS = 5
LEARNING_RATE = 2e-5

LOAD_CHECKPOINT = False
START_EPOCH = 0

CHECKPOINT_DIR = "../../models/pairwise_model_v3"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

TEST_SIZE = 0.1
IS_FULL_DATASET = True
DATASET_LIMIT = 1000
RANDOM_STATE = 42

# =====================================
# Load and process data
# =====================================

df = pd.read_csv(TRAIN_PATH)

train_df, val_df = preprocess_dataframe(
    df=df,
    test_size=TEST_SIZE,
    is_full_dataset=IS_FULL_DATASET,
    dataset_limit=DATASET_LIMIT,
    random_state=RANDOM_STATE
)

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

pairwise_model = PairwisePreferenceModel(backbone).to(device)

optimizer = torch.optim.AdamW(
    pairwise_model.parameters(),
    lr=LEARNING_RATE
)


# =====================================
# Tokenize ONCE
# =====================================

train_loader = tokenize_dataframe_train(
    train_df,
    tokenizer,
    MAX_LENGTH,
    BATCH_SIZE
)

# =====================================
# Load checkpoint
# =====================================

if LOAD_CHECKPOINT:
    checkpoint_path = f"{CHECKPOINT_DIR}/epoch_{START_EPOCH}.pt"

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    pairwise_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    print(f"Loaded checkpoint from: {checkpoint_path}")
else:
    START_EPOCH = 0


# =====================================
# Training
# =====================================

train_losses = []
train_steps = []


for i in range(NUM_EPOCHS):
    epoch = START_EPOCH + i

    pairwise_model.train()

    running_loss = 0.0
    num_examples = 0

    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{NUM_EPOCHS+START_EPOCH}"
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

        logits = pairwise_model(
            input_ids_a=enc_a["input_ids"],
            attention_mask_a=enc_a["attention_mask"],
            input_ids_b=enc_b["input_ids"],
            attention_mask_b=enc_b["attention_mask"]
        )

        loss = F.cross_entropy(
            logits,
            labels
        )

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
        f"Epoch {epoch + 1}/{NUM_EPOCHS+START_EPOCH} "
        f"- Train Loss: {train_loss:.4f}"
    )

    checkpoint_path = (
        f"{CHECKPOINT_DIR}/epoch_{epoch + 1}.pt"
    )

    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": pairwise_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
        },
        checkpoint_path
    )

    print(f"Checkpoint saved to: {checkpoint_path}")