# run.py

import os
import random
import numpy as np
import torch

from config import *
from dataset import load_dataset, split_df
from model import load_model
from trainer import train_model
from transformers import logging
from evaluator import evaluate_and_save


logging.set_verbosity_error()


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on: ", device)

    for name in DATASETS:
        print(f"\nLoading Dataset: {name}")
        df = load_dataset(DATA_DIR, name)

        for split in ["random", "time"]:
            if name == "trec2007" and split == "time":
                continue

            print(f"\nSplit Mode: {split}")

            train_df, test_df = split_df(
                df, split=split, train_ratio=TRAIN_RATIO, seed=SEED
            )

            # Use 10% from train as validation
            v_size = max(1, int(len(train_df) * 0.1))
            val_df = train_df.sample(n=v_size, random_state=SEED)
            tr_df = train_df.drop(val_df.index)

            model, tokenizer = load_model(MODEL_NAME, device)

            train_model(
                model,
                tokenizer,
                tr_df["text_clean"].values,
                tr_df["label"].values,
                val_df["text_clean"].values,
                val_df["label"].values,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                lr=LEARNING_RATE,
                max_length=MAX_LENGTH,
                device=device,
            )

            output_dir = os.path.join(OUTPUT_DIR, name, split)
            os.makedirs(output_dir, exist_ok=True)

            evaluate_and_save(
                model, tokenizer, test_df, device, output_dir, tag=f"{name}_{split}"
            )


if __name__ == "__main__":
    main()