import os
import torch

# Paths - use __file__ to resolve relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # Go up two levels from code/roberta/
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed_data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "roberta_output")

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
