# extra_evals.py
# =====================================================
# Extra evaluations: cross-dataset + learning curve
# =====================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# -----------------------------------------------------
#  IMPORT main project functions & paths
# -----------------------------------------------------
from baselines import (
    DIR_EXP, DIR_SUM,
    standardize_and_save,
    run_baselines
)

# Load all datasets (same as your main.py)
data = standardize_and_save()

# Make new directories
DIR_CD = os.path.join(DIR_EXP, "cross_dataset")
DIR_LC = os.path.join(DIR_EXP, "learning_curve")
os.makedirs(DIR_CD, exist_ok=True)
os.makedirs(DIR_LC, exist_ok=True)


# =====================================================
#   UNIVERSAL TRAIN/EVAL FUNCTION
# =====================================================
def train_eval_from_splits(train_df, test_df, tag, split="custom", cm_dir=DIR_EXP):

    train_df = train_df.dropna(subset=["text_clean", "label"]).copy()
    test_df  = test_df.dropna(subset=["text_clean", "label"]).copy()

    Xtr = train_df["text_clean"].values
    ytr = train_df["label"].astype(int).values
    Xte = test_df["text_clean"].values
    yte = test_df["label"].astype(int).values

    # TF–IDF
    vec = TfidfVectorizer(min_df=2, max_df=0.95, ngram_range=(1, 2))
    Xtrv = vec.fit_transform(Xtr)
    Xtev = vec.transform(Xte)

    models = {
        "NaiveBayes": MultinomialNB(),
        "LogReg": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "LinearSVM": LinearSVC(class_weight="balanced"),
    }

    rows = []

    for name, model in models.items():
        model.fit(Xtrv, ytr)
        yhat = model.predict(Xtev)

        acc = accuracy_score(yte, yhat)
        p, r, f1, _ = precision_recall_fscore_support(
            yte, yhat, average="binary", zero_division=0
        )

        rows.append({
            "model": name,
            "tag": tag,
            "split": split,
            "acc": acc,
            "prec": p,
            "recall": r,
            "f1": f1
        })

        # Confusion Matrix
        cm = confusion_matrix(yte, yhat, labels=[0,1])
        fig, ax = plt.subplots()
        ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{tag} | {name}")
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center")

        out = os.path.join(cm_dir, f"cm_{tag}_{name}.png")
        plt.tight_layout()
        plt.savefig(out, dpi=160)
        plt.close()
        print(f"[cross/lc] Saved CM → {out}")

    return pd.DataFrame(rows)


# =====================================================
#   1. CROSS DATASET EVAL
# =====================================================
def run_cross_dataset_eval(data_dict):

    datasets = ["spam_assassin", "enron", "trec2007"]
    all_rows = []

    for train_name in datasets:
        train_df = data_dict[train_name]

        for test_name in datasets:
            if train_name == test_name:
                continue

            test_df = data_dict[test_name]
            tag = f"train_{train_name}_test_{test_name}"

            print(f"\n=== Cross dataset: {tag} ===")

            res = train_eval_from_splits(
                train_df,
                test_df,
                tag=tag,
                split="cross",
                cm_dir=DIR_CD
            )
            all_rows.append(res)

    result = pd.concat(all_rows, ignore_index=True)
    out = os.path.join(DIR_SUM, "cross_dataset_metrics.csv")
    result.to_csv(out, index=False)

    print(f"\nSaved cross-dataset results → {out}")
    return result


# =====================================================
#   2. LEARNING CURVE EVAL
# =====================================================
from sklearn.model_selection import train_test_split

def run_learning_curve(df, tag_prefix):

    fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    all_rows = []

    # fix one random split
    df = df.dropna(subset=["text_clean","label"]).copy()
    train_full, test_set = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )

    for frac in fractions:
        # stratified down-sample
        train_frac = train_full.groupby("label", group_keys=False).apply(
            lambda g: g.sample(max(1, int(len(g) * frac)), random_state=42)
        )

        tag = f"{tag_prefix}_frac_{int(frac*100)}"
        print(f"\n=== Learning curve: {tag} ===")

        res = train_eval_from_splits(
            train_frac,
            test_set,
            tag=tag,
            split="learning_curve",
            cm_dir=DIR_LC
        )

        res["train_frac"] = frac
        res["n_train"] = len(train_frac)
        res["n_test"] = len(test_set)
        all_rows.append(res)

    result = pd.concat(all_rows, ignore_index=True)
    out = os.path.join(DIR_SUM, f"learning_curve_{tag_prefix}.csv")
    result.to_csv(out, index=False)

    print(f"\nSaved learning-curve results → {out}")
    return result


# =====================================================
#   MAIN ENTRY
# =====================================================
if __name__ == "__main__":

    print(">>> Running baseline (random + time-aware)…")
    run_baselines(data)

    print("\n>>> Running cross-dataset evaluation…")
    run_cross_dataset_eval(data)

    print("\n>>> Running learning curve evaluation…")
    for key in ["spam_assassin", "enron", "trec2007", "combined"]:
        print(f"\n--- Learning curve on {key} ---")
        run_learning_curve(data[key], tag_prefix=key)

    print("\nAll extra evaluations completed.")
