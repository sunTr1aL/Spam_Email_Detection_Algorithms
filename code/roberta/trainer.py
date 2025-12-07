import torch

from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from config import *


class EmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts, truncation=True, padding=True, max_length=MAX_LENGTH
        )
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def get_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(DEVICE)
    return model, tokenizer


def train_engine(train_df, val_df, output_tag):
    """
    Main training loop with Mixed Precision and Weight Decay.
    Returns: The best trained model (loaded with best weights) and tokenizer.
    """
    model, tokenizer = get_model_and_tokenizer()

    # Prepare Datasets
    train_ds = EmailDataset(
        train_df["text_clean"].tolist(), train_df["label"].tolist(), tokenizer
    )
    val_ds = EmailDataset(
        val_df["text_clean"].tolist(), val_df["label"].tolist(), tokenizer
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )

    # Optimizer & Scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)

    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # Mixed Precision Scaler
    scaler = GradScaler(device=DEVICE_STR)

    best_loss = float("inf")
    best_state = None

    print(
        f"[{output_tag}] Start Training: {len(train_df)} train, {len(val_df)} val samples."
    )

    for _ in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            with autocast(device_type=DEVICE_STR):
                outputs = model(**batch)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                with autocast(device_type=DEVICE_STR):
                    outputs = model(**batch)
                    val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_state = model.state_dict().copy()

    # Load best model
    if best_state:
        model.load_state_dict(best_state)

    return model, tokenizer
