# -*- coding: utf-8 -*-
"""
Email Spam Baselines: load → clean → standardize → model → evaluate
Folder layout:
data/
  raw/            # 原始CSV（输入）
  processed/      # 清洗后CSV（输出）
  eda/            # 分析图表
  experiments/
    random_split/ # 随机切分的混淆矩阵
    time_split/   # 时间切分的混淆矩阵
    summary/      # 指标汇总CSV等

"""

import os, re, warnings, json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from typing import Dict
from bs4 import BeautifulSoup

# ========= ROOT & FOLDERS =========
DATA_ROOT = r"D:\cs 410\final project\data"

DIR_RAW   = os.path.join(DATA_ROOT, "raw")
DIR_PROC  = os.path.join(DATA_ROOT, "processed")
DIR_EDA   = os.path.join(DATA_ROOT, "eda")
DIR_EXP   = os.path.join(DATA_ROOT, "experiments")
DIR_RS    = os.path.join(DIR_EXP, "random_split")
DIR_TS    = os.path.join(DIR_EXP, "time_split")
DIR_SUM   = os.path.join(DIR_EXP, "summary")

for d in [DIR_RAW, DIR_PROC, DIR_EDA, DIR_EXP, DIR_RS, DIR_TS, DIR_SUM]:
    os.makedirs(d, exist_ok=True)


def _pick_file(fname: str) -> str:
    p1 = os.path.join(DIR_RAW, fname)
    p2 = os.path.join(DATA_ROOT, fname)
    if os.path.exists(p1): return p1
    if os.path.exists(p2): return p2
    raise FileNotFoundError(f"File not found：{fname}\nput into {DIR_RAW} or {DATA_ROOT}")

FILES = {
    "spam_assassin": _pick_file("spam_assassin.csv"),
    "enron":         _pick_file("enron_spam_data.csv") if os.path.exists(os.path.join(DIR_RAW,"enron_spam_data.csv")) or os.path.exists(os.path.join(DATA_ROOT,"enron_spam_data.csv"))
                     else _pick_file("enron_clean.csv"),
    "trec2007":      _pick_file("trec_2007_data.csv"),
}

# ========= SWITCHES =========
RUN_EDA        = True
RUN_BASELINES  = True
RUN_TIME_AWARE = True
RUN_PER_USER   = True

# ========= ROBUST CSV READER =========
def read_csv_robust(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8", on_bad_lines="skip", engine="python")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", on_bad_lines="skip", engine="python")

# ========= LABEL NORM =========
def norm_label(v):
    if pd.isna(v): return None
    if isinstance(v, (int, float)): return int(v > 0)
    s = str(v).strip().lower()
    if s in ["1","spam","true","yes","y"]: return 1
    if s in ["0","ham","false","no","n"]:  return 0
    return 1 if "spam" in s else 0

# ========= CLEANING (NLTK or fallback) =========
import nltk, os as _os
_os.environ.setdefault("NLTK_DATA", r"D:\nltk_data")

def _ensure_nltk() -> bool:
    pkgs = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    ]
    ok = True
    for res_path, pkg in pkgs:
        try:
            nltk.data.find(res_path)
        except LookupError:
            try:
                nltk.download(pkg, quiet=False)
                nltk.data.find(res_path)
            except Exception:
                ok = False
    return ok

USE_NLTK = _ensure_nltk()

if USE_NLTK:
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    STOP = set(stopwords.words("english"))

    def clean_text(text: str) -> str:
        if not isinstance(text, str): return ""
        x = BeautifulSoup(text, "html.parser").get_text()
        x = re.sub(r"(https?://\S+|www\.\S+)", " ", x)
        x = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", " ", x)
        x = re.sub(r"[^a-zA-Z\s]", " ", x).lower()
        toks = [t for t in word_tokenize(x) if t not in STOP and len(t) > 2]
        return " ".join(toks)
else:
    BASIC_STOP = set("""
    a about above after again against all am an and any are as at be because been before
    being below between both but by can did do does doing down during each few for from further
    had has have having he her here hers herself him himself his how i if in into is it its itself
    just me more most my myself no nor not of off on once only or other our ours ourselves out
    over own same she should so some such than that the their theirs them themselves then there
    these they this those through to too under until up very was we were what when where which
    while who whom why will with you your yours yourself yourselves
    """.split())

    def clean_text(text: str) -> str:
        if not isinstance(text, str): return ""
        x = BeautifulSoup(text, "html.parser").get_text()
        x = re.sub(r"(https?://\S+|www\.\S+)", " ", x)
        x = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", " ", x)
        x = re.sub(r"[^A-Za-z\s]", " ", x).lower()
        toks = re.findall(r"[a-z]{3,}", x)
        toks = [t for t in toks if t not in BASIC_STOP]
        return " ".join(toks)

# ========= LOADERS =========
def parse_date_from_header(t):
    m = re.search(r"^Date:\s*(.+)$", str(t), flags=re.MULTILINE)
    return pd.to_datetime(m.group(1), errors="coerce") if m else pd.NaT

def load_spam_assassin(path: str) -> pd.DataFrame:
    df = read_csv_robust(path)
    assert set(["text","target"]).issubset(df.columns), f"spam_assassin columns={df.columns}"
    out = pd.DataFrame()
    out["source"] = "spam_assassin"
    out["text_raw"] = df["text"].astype(str)
    out["timestamp"] = df["text"].apply(parse_date_from_header)
    out["user_id"] = None
    out["label"] = df["target"].apply(norm_label)
    out["text_clean"] = out["text_raw"].apply(clean_text)
    out = out[out["text_clean"].str.len() > 0].reset_index(drop=True)
    return out

def load_enron(path: str) -> pd.DataFrame:
    df = read_csv_robust(path)
    need = ["Subject","Message","Spam/Ham","Date"]
    assert set(need).issubset(df.columns), f"enron columns={df.columns}"
    out = pd.DataFrame()
    out["source"] = "enron"
    subj = df["Subject"].fillna("").astype(str)
    body = df["Message"].fillna("").astype(str)
    out["text_raw"] = subj + "\n" + body
    out["timestamp"] = pd.to_datetime(df["Date"], errors="coerce")
    out["user_id"] = None
    out["label"] = df["Spam/Ham"].apply(norm_label)
    out["text_clean"] = out["text_raw"].apply(clean_text)
    out = out[out["text_clean"].str.len() > 0].reset_index(drop=True)
    return out

def load_trec2007(path: str) -> pd.DataFrame:
    df = read_csv_robust(path)
    need = ["label","subject","email_to","email_from","message"]
    assert set(need).issubset(df.columns), f"trec2007 columns={df.columns}"
    out = pd.DataFrame()
    out["source"] = "trec2007"
    subj = df["subject"].fillna("").astype(str)
    body = df["message"].fillna("").astype(str)
    out["text_raw"] = subj + "\n" + body
    out["timestamp"] = pd.NaT
    out["user_id"] = df["email_to"].astype(str)
    out["label"] = df["label"].apply(norm_label)
    out["text_clean"] = out["text_raw"].apply(clean_text)
    out = out[out["text_clean"].str.len() > 0].reset_index(drop=True)
    return out

LOADERS = {
    "spam_assassin": load_spam_assassin,
    "enron": load_enron,
    "trec2007": load_trec2007,
}

# ========= STANDARDIZE & SAVE =========
def standardize_and_save() -> Dict[str, pd.DataFrame]:
    data = {}
    for name, loader in LOADERS.items():
        df = loader(FILES[name])
        save_path = os.path.join(DIR_PROC, f"{name}_clean.csv")
        df.to_csv(save_path, index=False)
        print(f"{name} -> {save_path}  shape={df.shape}  spam_rate={df['label'].mean():.3f}")
        data[name] = df
    combined = pd.concat(list(data.values()), ignore_index=True)
    combined_path = os.path.join(DIR_PROC, "combined_clean.csv")
    combined.to_csv(combined_path, index=False)
    print("Merge completed ->", combined_path, combined.shape)
    data["combined"] = combined
    return data

# ========= EDA =========
def run_eda(data: Dict[str, pd.DataFrame]):
    import matplotlib.pyplot as plt

    rows = []
    for k in ["spam_assassin","enron","trec2007"]:
        d = data[k].copy()
        d["len"] = d["text_clean"].str.split().apply(len)
        rows.append({
            "source": k,
            "n_samples": len(d),
            "spam_rate": d["label"].mean(),
            "avg_len_spam": d.loc[d.label==1, "len"].mean(),
            "avg_len_ham":  d.loc[d.label==0, "len"].mean(),
        })
    eda_df = pd.DataFrame(rows).sort_values("source")
    eda_csv = os.path.join(DIR_EDA, "eda_summary.csv")
    eda_df.to_csv(eda_csv, index=False)
    print(f"Saved EDA summary -> {eda_csv}")

    fig, ax = plt.subplots()
    labels, vals = [], []
    for k in ["spam_assassin","enron","trec2007"]:
        d = data[k]
        vc = d["label"].value_counts(normalize=True)
        labels += [f"{k}-ham", f"{k}-spam"]
        vals   += [float(vc.get(0,0)), float(vc.get(1,0))]
    ax.bar(labels, vals)
    ax.set_ylabel("proportion")
    ax.set_title("Class proportion by dataset")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out = os.path.join(DIR_EDA, "class_proportion.png")
    plt.savefig(out, dpi=160)
    print(f" Saved: {out}")

# ========= MODELS & EVAL =========
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

def train_eval_one(
    df: pd.DataFrame,
    split: str = "random",   # 'random' or 'time'
    test_size: float = 0.2,
    seed: int = 42,
    tag: str = ""
) -> pd.DataFrame:
    d = df.dropna(subset=["text_clean","label"]).copy()
    if split == "time" and "timestamp" in d.columns and d["timestamp"].notna().any():
        d = d.sort_values("timestamp")
        n = len(d)
        n_train = max(1, int(n * (1 - test_size)))
        train, test = d.iloc[:n_train], d.iloc[n_train:]
        cm_dir = DIR_TS
    else:
        train, test = train_test_split(d, test_size=test_size, random_state=seed, stratify=d["label"])
        cm_dir = DIR_RS

    Xtr, ytr = train["text_clean"].values, train["label"].astype(int).values
    Xte, yte = test["text_clean"].values,  test["label"].astype(int).values

    vec = TfidfVectorizer(min_df=2, max_df=0.95, ngram_range=(1,2))
    Xtrv = vec.fit_transform(Xtr)
    Xtev = vec.transform(Xte)

    models = {
        "NaiveBayes": MultinomialNB(),
        "LogReg": LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=1),
        "LinearSVM": LinearSVC(class_weight="balanced"),
    }

    rows = []
    import matplotlib.pyplot as plt

    for name, model in models.items():
        model.fit(Xtrv, ytr)
        yhat = model.predict(Xtev)
        acc = accuracy_score(yte, yhat)
        p, r, f1, _ = precision_recall_fscore_support(yte, yhat, average="binary", zero_division=0)
        rows.append({"model": name, "split": split, "tag": tag, "acc": acc, "prec": p, "recall": r, "f1": f1})

        cm = confusion_matrix(yte, yhat, labels=[0,1])
        fig, ax = plt.subplots()
        ax.imshow(cm, cmap="Blues")
        ax.set_title(f"Confusion Matrix [{tag} | {name}]")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_xticks([0,1]); ax.set_xticklabels(["Ham(0)","Spam(1)"])
        ax.set_yticks([0,1]); ax.set_yticklabels(["Ham(0)","Spam(1)"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        plt.tight_layout()
        out = os.path.join(cm_dir, f"cm_{tag}_{name}.png")
        plt.savefig(out, dpi=160)
        plt.close(fig)
        print(f" Saved: {out}  | acc={acc:.3f}, f1={f1:.3f}, recall={r:.3f}")

    return pd.DataFrame(rows)

def run_baselines(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    all_rows = []
    for key in ["spam_assassin","enron","trec2007","combined"]:
        df = data[key]
        print(f"\n=== Baselines on [{key}] ===")
        res_rand = train_eval_one(df, split="random", test_size=0.2, tag=f"{key}_rand")
        all_rows.append(res_rand)
        if RUN_TIME_AWARE and key in ["spam_assassin","enron"]:
            if df["timestamp"].notna().any():
                res_time = train_eval_one(df, split="time", test_size=0.2, tag=f"{key}_time")
                all_rows.append(res_time)
            else:
                print(f"[{key}] No available timestamp")

    metrics = pd.concat(all_rows, ignore_index=True)
    metrics_path = os.path.join(DIR_SUM, "baseline_metrics.csv")
    metrics.to_csv(metrics_path, index=False)
    print(f"\n Saved metrics: {metrics_path}")

    pivot = metrics.pivot_table(index=["tag","model","split"], values=["acc","prec","recall","f1"])
    pivot_path = os.path.join(DIR_SUM, "baseline_metrics_pivot.csv")
    pivot.to_csv(pivot_path)
    print(f" Saved pivot: {pivot_path}")

    return metrics

# ========= PER-USER =========
def per_user_stats(df: pd.DataFrame, min_samples: int = 10) -> pd.DataFrame:
    if "user_id" not in df.columns or df["user_id"].isna().all():
        print("No user_id available; skip per-user analysis.")
        return pd.DataFrame()

    from sklearn.model_selection import train_test_split
    d = df.dropna(subset=["text_clean","label"]).copy()
    if d["user_id"].nunique() < 2:
        print("Only one user_id; skip.")
        return pd.DataFrame()

    train, test = train_test_split(d, test_size=0.2, random_state=42, stratify=d["label"])

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    vec = TfidfVectorizer(min_df=2, max_df=0.95, ngram_range=(1,2))
    Xtrv = vec.fit_transform(train["text_clean"])
    Xtev = vec.transform(test["text_clean"])

    clf = MultinomialNB()
    clf.fit(Xtrv, train["label"].astype(int).values)

    test = test.copy()
    test["pred"] = clf.predict(Xtev)

    rows = []
    for uid, g in test.groupby("user_id"):
        if len(g) < min_samples:
            continue
        p, r, f1, _ = precision_recall_fscore_support(g["label"].astype(int), g["pred"], average="binary", zero_division=0)
        rows.append({"user_id": uid, "n": len(g), "prec": p, "recall": r, "f1": f1})
    res = pd.DataFrame(rows).sort_values("f1", ascending=False)

    out = os.path.join(DIR_SUM, "per_user_stats.csv")
    res.to_csv(out, index=False)
    print(f" Saved per-user stats: {out}  (users kept: {len(res)})")
    if len(res):
        print("Per-user F1 variance:", res["f1"].var())
    return res

# ========= MAIN =========
def main():
    print("== Standardizing and cleaning datasets ==")
    data = standardize_and_save()

    if RUN_EDA:
        run_eda(data)

    if RUN_BASELINES:
        run_baselines(data)

    if RUN_PER_USER and "trec2007" in data:
        per_user_stats(data["trec2007"], min_samples=10)

    print("\nAll done.")

if __name__ == "__main__":
    main()
