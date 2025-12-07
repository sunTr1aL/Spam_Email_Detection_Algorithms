# trainer.py

import torch


from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorWithPadding, get_linear_schedule_with_warmup


class EmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
        )
        enc["labels"] = self.labels[idx]
        return enc


def train_model(
    model,
    tokenizer,
    text_train,
    labels_train,
    text_val,
    labels_val,
    epochs,
    batch_size,
    lr,
    max_length,
    device,
):

    train_ds = EmailDataset(text_train, labels_train, tokenizer, max_length)
    val_ds = EmailDataset(text_val, labels_val, tokenizer, max_length)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collator
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collator
    )

    optim = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=int(0.1 * len(train_loader) * epochs),
        num_training_steps=len(train_loader) * epochs,
    )
    best_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for batch in train_loader:
            batch = {
                k: (
                    torch.tensor(v).to(device)
                    if not isinstance(v, torch.Tensor)
                    else v.to(device)
                )
                for k, v in batch.items()
            }

            out = model(**batch)
            loss = out.loss
            loss.backward()
            optim.step()
            scheduler.step()
            optim.zero_grad()
            total += loss.item()

        avg = total / len(train_loader)
        print(f"  Epoch {epoch+1}/{epochs}: train loss={avg:.4f}")

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {
                    k: (
                        torch.tensor(v).to(device)
                        if not isinstance(v, torch.Tensor)
                        else v.to(device)
                    )
                    for k, v in batch.items()
                }
                out = model(**batch)
                val_total += out.loss.item()

        val_avg = val_total / len(val_loader)
        print(f"  val loss={val_avg:.4f}")

        if val_avg < best_loss:
            best_loss = val_avg
            best_state = model.state_dict().copy()

    if best_state:
        model.load_state_dict(best_state)