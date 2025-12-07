# config.py

DATA_DIR = "./data/processed"
OUTPUT_DIR = "./experiments_roberta"

DATASETS = ["spam_assassin", "enron", "trec2007"]

MODEL_NAME = "roberta-base"
SEED = 42
TRAIN_RATIO = 0.8
EPOCHS = 3
BATCH_SIZE = 24
LEARNING_RATE = 2e-5
MAX_LENGTH = 256