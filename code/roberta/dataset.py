import os
import pandas as pd

from sklearn.model_selection import train_test_split

from config import SEED


TRUE_LABEL_SET = {"1", "spam", "true", "yes", "y", "1.0"}
FALSE_LABEL_SET = {"0", "ham", "false", "no", "n", "0.0"}


def normalize_label(v):
    if pd.isna(v):
        return None
    s = str(v).strip().lower()
    if s in TRUE_LABEL_SET:
        return 1
    if s in FALSE_LABEL_SET:
        return 0
    return None


def load_dataset(data_dir, name):
    path = os.path.join(data_dir, f"{name}_clean.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_csv(path, engine="python")
    df["label"] = df["label"].apply(normalize_label)
    df = df.dropna(subset=["label", "text_clean"])
    df["label"] = df["label"].astype(int)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df.reset_index(drop=True)


def get_splits(df, mode="random", train_ratio=0.8):
    if mode == "time":
        if "timestamp" not in df.columns or df["timestamp"].isnull().all():
            raise ValueError(f"No available timestamp")

        df_sorted = df.sort_values("timestamp").reset_index(drop=True)
        split_idx = int(len(df_sorted) * train_ratio)
        return df_sorted[:split_idx], df_sorted[split_idx:]

    elif mode == "random":
        return train_test_split(
            df, train_size=train_ratio, random_state=SEED, stratify=df["label"]
        )

    else:
        raise ValueError(f"Unknown split mode: {mode}")
