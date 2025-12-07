# model.py

from transformers import RobertaForSequenceClassification, AutoTokenizer


def load_model(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)
    return model, tokenizer