# LLM Classification Finetuning

This project is my solution for the **LLM Classification Finetuning** competition on Kaggle.

The goal is to predict which of two LLM responses a user prefers:

- `0` — Response A
- `1` — Response B
- `2` — Tie

## Approach

I started with a DeBERTa-v3-small model and treated the task as a preference prediction problem.

Instead of giving each response a score separately, I built a **pairwise model**:

    Response A ──> DeBERTa ──> hA ──┐
                                     ├──> Pairwise features ──> MLP ──> A/B/Tie
    Response B ──> DeBERTa ──> hB ──┘

For each response, I use attention pooling to get one vector. I then combine the two vectors using:

- `hA`
- `hB`
- `|hA - hB|`
- `hA * hB`

These features are passed to a small MLP that predicts the three classes.

## Data

I use a 90/10 train-validation split.

The dataset is tokenized before training so that tokenization does not happen for every training step.

For long conversations, I use a head/tail truncation strategy with a maximum length of 512 tokens.

## Results

With 10,000 training examples, the pairwise model reached:

    Epoch 1: Val Loss = 1.0158
    Epoch 2: Val Loss = 1.0090
    Epoch 3: Val Loss = 1.0789
    Epoch 4: Val Loss = 1.3267
    Epoch 5: Val Loss = 1.7226

The best validation result was **1.0090 at epoch 2**.

The model started to overfit after the second epoch, so I use the validation loss to choose the checkpoint rather than simply training for more epochs.

## Project Structure

    llm-classification-finetuning/
    ├── data/
    ├── model/
    │   ├── pairwise_model_v3/
    │   │   ├── data.py
    │   │   ├── model.py
    │   │   └── train.py
    │   ├── reward_model_v2/
    │   │   │   ├── data.py
    │   │   ├── model.py
    │   │   └── train.py
    ├── models/
    ├── notebooks/
    └── README.md

## Next Steps

Some things I plan to try next:

- DeBERTa-v3-base
- A/B response swapping to reduce position bias
- Label smoothing
- Better handling of long conversations
- Larger LLMs with LoRA/QLoRA

The main goal is to improve the validation log loss while keeping the model small enough to train and experiment with locally.