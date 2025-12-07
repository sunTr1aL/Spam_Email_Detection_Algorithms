# dataset.py

import os
import pandas as pd


def normalize_label(v):
    if pd.isna(v):
        return None
    if isinstance(v, (int, float)):
        return int(v > 0)
    s = str(v).strip().lower()
    if s in {"1", "spam", "true", "yes", "y"}:
        return 1
    if s in {"0", "ham", "false", "no", "n"}:
        return 0
    return None


def load_dataset(data_dir, name) -> pd.DataFrame:
    path = os.path.join(data_dir, f"{name}_clean.csv")
    df = pd.read_csv(path, engine="python")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    df["label"] = df["label"].apply(normalize_label)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    df = df.dropna(subset=["text_clean"])
    return df


def split_df(df, split, train_ratio, seed):
    if split == "time":
        df_sorted = df.sort_values("timestamp").reset_index(drop=True)
        n_train = int(len(df_sorted) * train_ratio)
        return df_sorted[:n_train], df_sorted[n_train:]

    elif split == "random":
        from sklearn.model_selection import train_test_split

        train, test = train_test_split(
            df, train_size=train_ratio, random_state=seed, stratify=df["label"]
        )
        return train, test

    raise RuntimeError('Split should be "time" or "random"')