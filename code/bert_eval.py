# -*- coding: utf-8 -*-
"""
DistilBERT + LoRA experiments on email spam datasets.

Experiments:
  1) Baseline (random split) on each dataset:
       - spam_assassin, enron, trec2007, combined
  2) Time-aware: train on earlier emails, test on later emails
       - spam_assassin, enron (where timestamps exist)
  3) Cross-dataset: train on one dataset, test on the other two
  4) Training-size (combined):
       - Hold out 30% of the entire dataset as test (fixed)
       - Train on 10%, 20%, ..., 70% of the entire dataset

Input (processed data):
  data/processed_data/
      spam_assassin_clean.csv
      enron_clean.csv
      trec2007_clean.csv
      combined_clean.csv

Output (all under DATA_ROOT/bert_output):
  bert_output/
      random_split/         # confusion matrices for baseline
      time_split/           # confusion matrices for time-aware
      cross_dataset/        # confusion matrices for cross-dataset
      training_size/        # confusion matrices + curves for training-size
      runs/                 # HF Trainer outputs
      summary/
          distilbert_lora_random_split.csv
          distilbert_lora_time_split.csv
          distilbert_lora_cross_dataset.csv
          distilbert_lora_training_size.csv
          distilbert_lora_all_experiments.csv
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
import matplotlib.pyplot as plt

from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType


# === Paths ===
THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # .../code
PROJECT_ROOT = os.path.dirname(THIS_DIR)               # repo root

DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
DIR_PROC  = os.path.join(DATA_ROOT, "processed_data")

BERT_ROOT = os.path.join(PROJECT_ROOT, "bert_output")
DIR_RS    = os.path.join(BERT_ROOT, "random_split")     # baseline (random split)
DIR_TS    = os.path.join(BERT_ROOT, "time_split")       # time-aware
DIR_CD    = os.path.join(BERT_ROOT, "cross_dataset")    # cross-dataset
DIR_TRS   = os.path.join(BERT_ROOT, "training_size")    # training-size
DIR_SUM   = os.path.join(BERT_ROOT, "summary")          # all BERT metrics
DIR_RUNS  = os.path.join(BERT_ROOT, "runs")             # HF Trainer outputs

for d in [BERT_ROOT, DIR_RS, DIR_TS, DIR_CD, DIR_TRS, DIR_SUM, DIR_RUNS]:
    os.makedirs(d, exist_ok=True)


# ========= GLOBAL CONFIG =========
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 256
BATCH_SIZE = 8
EPOCHS = 3
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)


# ========= DATA LOADING & TOKENIZATION =========
def load_clean_csv(name: str) -> pd.DataFrame:
    """
    Load <name>_clean.csv from processed_data.
    Requires 'text_clean' and 'label'; parses 'timestamp' if present.
    """
    path = os.path.join(DIR_PROC, f"{name}_clean.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    if "text_clean" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{path} must contain 'text_clean' and 'label' columns")

    df = df.dropna(subset=["text_clean", "label"])
    df = df[df["text_clean"].astype(str).str.len() > 0].copy()
    df["label"] = df["label"].astype(int)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    return df


def tokenize_dataset(ds: Dataset) -> Dataset:
    """Tokenize a HF Dataset with columns: 'text_clean', 'label'."""
    return ds.map(
        lambda batch: tokenizer(
            batch["text_clean"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
        ),
        batched=True,
    )


# ========= MODEL (DistilBERT + LoRA) =========
from peft import LoraConfig, get_peft_model, TaskType

def make_lora_model(num_labels: int = 2):
    base_model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
    )

    # DistilBERT attention layers use q_lin / k_lin / v_lin / out_lin
    # We'll add LoRA adapters to q_lin and v_lin.
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_lin", "v_lin"],  # <-- important line
    )

    model = get_peft_model(base_model, lora_config)

    # Print how many parameters are trainable vs frozen under LoRA
    model.print_trainable_parameters()

    return model



# ========= VISUALIZATION =========
def save_confusion_matrix(cm: np.ndarray, tag: str, cm_dir: str):
    """Save a confusion matrix PNG for Ham(0)/Spam(1), in the given directory."""
    fig, ax = plt.subplots()
    ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Confusion Matrix [{tag} | DistilBERT+LoRA]")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Ham(0)", "Spam(1)"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Ham(0)", "Spam(1)"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    plt.tight_layout()
    out_path = os.path.join(cm_dir, f"cm_{tag}_DistilBERT_LoRA.png")
    plt.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"  Saved confusion matrix -> {out_path}")


# ========= CORE TRAIN + EVAL WRAPPER =========
def train_eval_lora(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tag: str,
    exp_group: str,           # 'baseline' | 'time' | 'cross' | 'train_size'
    split: str,               # 'random'  | 'time' | 'cross' | 'train_size'
    train_dataset: str,
    test_dataset: str,
    train_frac_pool: float,   # fraction of the "train pool" used (0-1)
    train_frac_total: float,  # fraction of the TOTAL source dataset used (0-1)
    cm_dir: str,
):
    """Train DistilBERT+LoRA on train_df, evaluate on test_df and return metrics row."""

    print(f"\n--- DistilBERT+LoRA [{exp_group} | {tag}] ---")
    print(f"  train samples = {len(train_df)}, test samples = {len(test_df)}")
    print(f"  train_frac_pool={train_frac_pool:.3f}, train_frac_total={train_frac_total:.3f}")

    # Make HF Datasets
    train_ds = Dataset.from_pandas(
        train_df[["text_clean", "label"]].reset_index(drop=True)
    )
    test_ds = Dataset.from_pandas(
        test_df[["text_clean", "label"]].reset_index(drop=True)
    )

    train_ds = tokenize_dataset(train_ds)
    test_ds = tokenize_dataset(test_ds)

    train_ds = train_ds.rename_column("label", "labels")
    test_ds = test_ds.rename_column("label", "labels")

    train_ds.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    test_ds.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    model = make_lora_model(num_labels=2)

    args = TrainingArguments(
        output_dir=os.path.join(DIR_RUNS, tag),
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=100,
        save_strategy="no",
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
    )

    trainer.train()

    preds = trainer.predict(test_ds)
    y_pred = np.argmax(preds.predictions, axis=1)
    y_true = preds.label_ids

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    save_confusion_matrix(cm, tag, cm_dir)

    print(f"  acc={acc:.4f}  prec={prec:.4f}  rec={rec:.4f}  f1={f1:.4f}")

    return {
        "model": "DistilBERT_LoRA",
        "exp_group": exp_group,          # baseline / time / cross / train_size
        "split": split,
        "tag": tag,
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "train_frac_pool": float(train_frac_pool),
        "train_frac_total": float(train_frac_total),
        "acc": acc,
        "prec": prec,
        "recall": rec,
        "f1": f1,
    }


# ========= EXPERIMENT 1: BASELINE (RANDOM SPLIT) =========
def run_baseline_experiments(data_dict, test_size: float = 0.2):
    """Random train/test split on each dataset separately."""
    results = []
    for name in ["spam_assassin", "enron", "trec2007", "combined"]:
        df = data_dict[name]
        source_n = len(df)

        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df["label"],
            random_state=SEED,
        )

        train_frac_pool = len(train_df) / source_n  # here pool == total
        train_frac_total = train_frac_pool

        tag = f"{name}_rand"
        row = train_eval_lora(
            train_df=train_df,
            test_df=test_df,
            tag=tag,
            exp_group="baseline",
            split="random",
            train_dataset=name,
            test_dataset=name,
            train_frac_pool=train_frac_pool,
            train_frac_total=train_frac_total,
            cm_dir=DIR_RS,
        )
        results.append(row)
    return results


# ========= EXPERIMENT 2: CROSS-DATASET =========
def run_cross_dataset_experiments(data_dict):
    """Train on one dataset, test on the other two."""
    results = []
    names = ["spam_assassin", "enron", "trec2007"]

    for train_name in names:
        train_source = data_dict[train_name]
        source_n = len(train_source)
        train_df = train_source  # use full dataset
        train_frac_pool = len(train_df) / source_n  # == 1.0
        train_frac_total = train_frac_pool

        for test_name in names:
            if test_name == train_name:
                continue
            test_df = data_dict[test_name]
            tag = f"cross_{train_name}_to_{test_name}"
            row = train_eval_lora(
                train_df=train_df,
                test_df=test_df,
                tag=tag,
                exp_group="cross",
                split="cross",
                train_dataset=train_name,
                test_dataset=test_name,
                train_frac_pool=train_frac_pool,
                train_frac_total=train_frac_total,
                cm_dir=DIR_CD,
            )
            results.append(row)
    return results


# ========= EXPERIMENT 3: TIME-AWARE =========
def run_time_aware_experiments(data_dict, test_size: float = 0.2):
    """Train on earlier emails, test on later emails."""
    results = []
    for name in ["spam_assassin", "enron"]:
        if name not in data_dict:
            continue
        df = data_dict[name]
        if "timestamp" not in df.columns:
            continue

        d = df.dropna(subset=["text_clean", "label", "timestamp"]).copy()
        if len(d) < 10:
            continue

        d = d.sort_values("timestamp")
        source_n = len(d)

        n_train = max(1, int(source_n * (1 - test_size)))
        train_df = d.iloc[:n_train].copy()
        test_df = d.iloc[n_train:].copy()
        if len(test_df) < 10:
            continue

        train_frac_pool = len(train_df) / source_n
        train_frac_total = train_frac_pool

        tag = f"{name}_time"
        row = train_eval_lora(
            train_df=train_df,
            test_df=test_df,
            tag=tag,
            exp_group="time",
            split="time",
            train_dataset=name,
            test_dataset=name,
            train_frac_pool=train_frac_pool,
            train_frac_total=train_frac_total,
            cm_dir=DIR_TS,
        )
        results.append(row)

    return results


# ========= EXPERIMENT 4: TRAINING-SIZE =========
def run_training_size_experiments(combined_df: pd.DataFrame, test_fraction: float = 0.3):
    """
    Training-size experiment (combined dataset):

      - Fix 30% of the *entire* dataset as the test set.
      - Train on 10%, 20%, ..., 70% of the *entire* dataset.
        (All training subsets are sampled from the remaining 70%.)

      Example:
        N = 100k
        test = 30k (fixed)
        train sizes: 10k, 20k, ..., 70k (all drawn from the same 70k pool)
    """
    results = []

    df = combined_df.dropna(subset=["text_clean", "label"]).copy()
    df["label"] = df["label"].astype(int)
    total_n = len(df)

    # 1) Fix 30% of ENTIRE dataset as test
    train_pool, test_df = train_test_split(
        df,
        test_size=test_fraction,
        stratify=df["label"],
        random_state=SEED,
    )

    pool_n = len(train_pool)
    print(f"\n[Training-size] total={total_n}, train_pool={pool_n}, test={len(test_df)}")

    # 2) Fractions of the TOTAL dataset we want as training
    max_train_frac = 1.0 - test_fraction  # here 0.7
    total_fracs = [round(f, 1) for f in np.arange(0.1, max_train_frac + 0.05, 0.1)]
    total_fracs = [f for f in total_fracs if f <= max_train_frac + 1e-8]

    for frac_total in total_fracs:
        # Requested number of training samples measured against entire dataset
        n_train = int(round(frac_total * total_n))

        if n_train >= pool_n:
            print(f"[Training-size] Skipping frac_total={frac_total:.2f} "
              f"because n_train={n_train} >= pool_n={pool_n}")
            continue
        
        # Fraction of the *train pool* we actually use
        train_frac_pool = n_train / pool_n if pool_n > 0 else 0.0
        train_frac_total = n_train / total_n if total_n > 0 else 0.0

        # Stratified subsample from train_pool
        sub_train, _ = train_test_split(
            train_pool,
            train_size=n_train,
            stratify=train_pool["label"],
            random_state=SEED,
        )

        pct_total = int(round(train_frac_total * 100))
        tag = f"combined_trainsize_{pct_total}"

        row = train_eval_lora(
            train_df=sub_train,
            test_df=test_df,
            tag=tag,
            exp_group="train_size",
            split="train_size",
            train_dataset="combined",
            test_dataset="combined",
            train_frac_pool=train_frac_pool,
            train_frac_total=train_frac_total,
            cm_dir=DIR_TRS,
        )
        results.append(row)

    return results


# ========= PLOTTING FOR TRAINING-SIZE =========
def plot_training_size_curves(train_size_rows):
    """Create convenience plots of performance vs training fraction."""
    if not train_size_rows:
        print("[Training-size] No rows to plot.")
        return

    df = pd.DataFrame(train_size_rows).copy()
    df = df.sort_values("train_frac_total")

    xs = df["train_frac_total"] * 100.0  # percentage of entire dataset

    # Plot F1 and accuracy
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].plot(xs, df["f1"], marker="o")
    ax[0].set_title("F1 vs Training Size (combined)")
    ax[0].set_xlabel("% of entire dataset used for training")
    ax[0].set_ylabel("F1 score")
    ax[0].grid(True, linestyle="--", alpha=0.5)

    ax[1].plot(xs, df["acc"], marker="o", color="orange")
    ax[1].set_title("Accuracy vs Training Size (combined)")
    ax[1].set_xlabel("% of entire dataset used for training")
    ax[1].set_ylabel("Accuracy")
    ax[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(DIR_TRS, "training_size_curves.png")
    plt.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[Training-size] Saved curves -> {out_path}")


# ========= MAIN =========
def main():
    # Load processed datasets
    data = {
        "spam_assassin": load_clean_csv("spam_assassin"),
        "enron": load_clean_csv("enron"),
        "trec2007": load_clean_csv("trec2007"),
        "combined": load_clean_csv("combined"),
    }

    # 1) Baseline (random split)
    baseline_rows = run_baseline_experiments(data)
    pd.DataFrame(baseline_rows).to_csv(
        os.path.join(DIR_SUM, "distilbert_lora_random_split.csv"), index=False
    )

    # 2) Time-aware
    time_rows = run_time_aware_experiments(data)
    pd.DataFrame(time_rows).to_csv(
        os.path.join(DIR_SUM, "distilbert_lora_time_split.csv"), index=False
    )

    # 3) Cross-dataset
    cross_rows = run_cross_dataset_experiments(data)
    pd.DataFrame(cross_rows).to_csv(
        os.path.join(DIR_SUM, "distilbert_lora_cross_dataset.csv"), index=False
    )

    # 4) Training-size
    train_size_rows = run_training_size_experiments(data["combined"])
    pd.DataFrame(train_size_rows).to_csv(
        os.path.join(DIR_SUM, "distilbert_lora_training_size.csv"), index=False
    )
    plot_training_size_curves(train_size_rows)

    # Combined CSV
    all_rows = baseline_rows + time_rows + cross_rows + train_size_rows
    pd.DataFrame(all_rows).to_csv(
        os.path.join(DIR_SUM, "distilbert_lora_all_experiments.csv"), index=False
    )

    print("\nAll DistilBERT+LoRA experiments finished.")


if __name__ == "__main__":
    main()
