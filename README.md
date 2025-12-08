# CS410 Final Project – Email Spam Detection

This repository implements an email spam detection system and compares:

1. **Traditional machine learning baselines** - TF–IDF + Multinomial Naive Bayes  
   - TF–IDF + Logistic Regression  
   - TF–IDF + Linear SVM  

2. **Modern Transformer-based models** - DistilBERT fine-tuned with **LoRA** (parameter-efficient finetuning)

The goal is to evaluate how well different modeling approaches work on email spam datasets, and how much training data is needed to reach “good enough” performance.

---

## Datasets

All datasets are converted into a common schema:

- `text_raw`: original subject + body
- `text_clean`: cleaned text used for modeling
- `label`: `0 = ham`, `1 = spam`
- `timestamp`: parsed send time (if available)
- `user_id`: recipient (where available, e.g., TREC 2007)
- `source`: which dataset the row came from

### Datasets Used
1. **SpamAssassin** – classic spam filtering corpus  
2. **Enron Spam** – emails from the Enron corpus with spam labels  
3. **TREC 2007** – TREC 2007 Spam Track corpus  
4. **Combined** – union of the three datasets above

### Data Locations
**Raw CSVs (Input):**
data/raw_data/
    spam_assassin.csv
    enron_spam_data.csv  (or enron_clean.csv)
    trec_2007_data.csv

**Processed CSVs:**
data/processed_data/
    spam_assassin_clean.csv
    enron_clean.csv
    trec2007_clean.csv
    combined_clean.csv

## Project Structure

.
├── code/
│   ├── baselines.py     # traditional ML pipeline (TF-IDF + NB/LogReg/SVM)
│   ├── bert_eval.py     # DistilBERT + LoRA experiments
│   └── device.py        # optional helper for selecting device (GPU/CPU)
│
├── data/
│   ├── raw_data/        # original CSVs (input)
│   ├── processed_data/  # cleaned CSVs (output from baselines preprocessing)
│   └── eda/             # EDA plots and summaries
│
├── traditional_output/  # results from traditional baselines
│   ├── random_split/    # confusion matrices for random train/test
│   ├── time_split/      # confusion matrices for time-aware splits
│   └── summary/         # metrics CSVs (accuracy, precision, recall, F1)
│
├── bert_output/         # results from DistilBERT + LoRA experiments
│   ├── random_split/    # confusion matrices for random splits
│   ├── time_split/      # confusion matrices for time-aware splits
│   ├── cross_dataset/   # confusion matrices for cross-dataset evaluation
│   ├── training_size/   # confusion matrices + training-size curves
│   ├── summary/         # metrics CSVs for all BERT experiments
│   └── runs/            # HuggingFace Trainer run artifacts (checkpoints, logs)
│
├── per_user_stats.csv   # per-user performance summary (traditional NB baseline)
├── LICENSE
└── README.md