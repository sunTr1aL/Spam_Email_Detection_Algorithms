import os
import torch

# Paths
DATA_DIR = os.path.abspath("../../data/processed_data")  # Project/data/processed_data/
OUTPUT_DIR = os.path.abspath("../../roberta_output")  # Project/roberta_output/

# Datasets
DATASETS = ["enron", "spam_assassin", "trec2007"]

# Hyperparameters
MODEL_NAME = "roberta-base"
SEED = 42
EPOCHS = 3
BATCH_SIZE = 128
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
MAX_LENGTH = 256
TRAIN_RATIO = 0.8

# Device
DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STR)

os.makedirs(OUTPUT_DIR, exist_ok=True)
