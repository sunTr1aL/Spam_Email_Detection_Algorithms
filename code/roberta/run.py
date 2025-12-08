import os
import numpy as np
import random
import torch

from transformers import logging

# suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.set_verbosity_error()

from config import *
from dataset import load_dataset, get_splits
from trainer import train_engine
from evaluator import evaluate_and_save


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_experiment(train_df, test_df, exp_path, tag):
    v_size = max(1, int(len(train_df) * 0.1))
    val_df = train_df.sample(n=v_size, random_state=SEED)
    tr_df = train_df.drop(val_df.index)

    model, tokenizer = train_engine(tr_df, val_df, tag)
    evaluate_and_save(model, tokenizer, test_df, exp_path, tag)

    del model
    del tokenizer
    torch.cuda.empty_cache()


def main():
    set_seed(SEED)
    print(f"Running RoBERTa Experiments on {DEVICE}")
    print(f"Data Dir: {DATA_DIR}")
    print(f"Output Dir: {OUTPUT_DIR}")

    # Load all datasets into memory
    dfs = {}
    for name in DATASETS:
        try:
            dfs[name] = load_dataset(DATA_DIR, name)
            print(f"Loaded {name}: {len(dfs[name])} records")
        except FileNotFoundError as e:
            print(e)
            continue

    # 1. & 2. In-Domain Experiments (Random & Time Split)
    print("\n========== Experiment 1 & 2: In-Domain (Random & Time) ==========")
    for name, df in dfs.items():
        for mode in ["random", "time"]:
            if name != "enron" and mode == "time":
                continue

            tag = f"{name}_{mode}"
            print(f"\n--- Processing {tag}")

            train_df, test_df = get_splits(df, mode=mode, train_ratio=0.8)
            if mode == "random":
                out_path = os.path.join(OUTPUT_DIR, "random_split")
            elif mode == "time":
                out_path = os.path.join(OUTPUT_DIR, "time_split")

            run_experiment(train_df, test_df, out_path, tag)

    # 3. Cross Dataset Experiments
    print("\n========== Experiment 3: Cross Dataset ==========")
    out_path = os.path.join(OUTPUT_DIR, "cross_dataset")

    for src_name, src_df in dfs.items():
        print(f"\nTraining on Source: {src_name}")

        v_size = int(len(src_df) * 0.1)
        val_df = src_df.sample(n=v_size, random_state=SEED)
        tr_df = src_df.drop(val_df.index)

        model, tokenizer = train_engine(tr_df, val_df, f"src_{src_name}")

        # Test on other datasets
        for tgt_name, tgt_df in dfs.items():
            if src_name == tgt_name:
                continue

            print(f"  Testing on Target: {tgt_name}")
            evaluate_and_save(
                model,
                tokenizer,
                tgt_df,
                out_path,
                tag=f"train_{src_name}_test_{tgt_name}",
            )

        del model
        torch.cuda.empty_cache()

    # 4. Learning Curve (10% - 70% Train)
    print("\n========== Experiment 4: Learning Curve (10% - 70%) ==========")
    out_path = os.path.join(OUTPUT_DIR, "learning_curve")
    ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    for name, df in dfs.items():
        print(f"\nDataset: {name}")
        for r in ratios:
            tag = f"{name}_ratio_{r}"
            train_df, test_df = get_splits(df, mode="random", train_ratio=r)

            print(f"  Ratio {r}: Train size {len(train_df)}, Test size {len(test_df)}")
            run_experiment(train_df, test_df, out_path, tag)


if __name__ == "__main__":
    main()
