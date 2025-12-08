import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from config import *


def evaluate_and_save(model, tokenizer, test_df, sub_dir, tag):
    """
    Evaluates model on test_df, saves metrics.csv and confusion_matrix.png
    """
    os.makedirs(sub_dir, exist_ok=True)

    model.eval()

    # Batch evaluation to avoid OOM on large test sets
    texts = test_df["text_clean"].tolist()
    labels = test_df["label"].tolist()

    all_preds = []
    all_probs = []

    # Simple batching for inference
    batch_size = BATCH_SIZE * 2
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            # Softmax output layer as requested
            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    # Save Metrics
    acc = accuracy_score(labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, all_preds, average="binary", zero_division=0
    )

    metrics_path = os.path.join(sub_dir, f"metrics_{tag}.csv")
    pd.DataFrame(
        [
            {
                "dataset": tag,
                "accuracy": acc,
                "precision": p,
                "recall": r,
                "f1": f1,
                "size": len(test_df),
            }
        ]
    ).to_csv(metrics_path, index=False)

    print(f"  -> {tag}: F1={f1:.4f}, Acc={acc:.4f} saved to {sub_dir}")

    # Plot Confusion Matrix
    cm = confusion_matrix(labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix: {tag}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(sub_dir, f"cm_{tag}.png"))
    plt.close()
