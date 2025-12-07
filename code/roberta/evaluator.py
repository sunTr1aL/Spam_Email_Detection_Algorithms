# evaluator.py

import torch
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


def evaluate_and_save(model, tokenizer, df_test, device, out_dir, tag):
    model.eval()
    texts = df_test["text_clean"].values
    labels = df_test["label"].astype(int).values

    preds, probs = [], []
    for text in texts:
        enc = tokenizer(text, truncation=True, max_length=256, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
            p = torch.softmax(out.logits, dim=1)
            prob = p[:, 1].item()
            pred = p.argmax(dim=1).item()
        preds.append(pred)
        probs.append(prob)

    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    cm = confusion_matrix(labels, preds)

    metrics = pd.DataFrame([{"acc": acc, "prec": p, "recall": r, "f1": f1}])
    metrics.to_csv(f"{out_dir}/metrics_{tag}.csv", index=False)

    fig, ax = plt.subplots()
    ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Confusion Matrix {tag}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    plt.savefig(f"{out_dir}/cm_{tag}.png")
    plt.close()