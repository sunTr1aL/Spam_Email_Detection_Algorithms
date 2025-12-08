📧 Spam Email Detection Project

This project is the GitHub repository for **CS410 Final Project (Group 52)** at UIUC.
---
#
## 🧠 Overview
#
This repository provides a full framework for **spam email detection** and experimentation. It includes:
#
- 📊 **Data Standardization** — unified preprocessing across major public datasets:
  - Enron Email Dataset
  - SpamAssassin
  - TREC 2007
#
- ⚙️ **Baseline Models** — classic machine learning benchmarks using TF–IDF:
  - Multinomial Naive Bayes
  - Logistic Regression
  - Linear SVM
#
- 🤖 **Transformer-Based Classifiers**
  - **DistilBERT + LoRA**
  - **RoBERTa**
#
- 📈 **Automated Reporting**
  - Evaluation metrics (accuracy, precision, recall, F1) as CSV files.
  - Visual **confusion matrices** and training-size curves.
---
#
## 🧩 Datasets
#
All datasets are standardized to a common schema with (at least) the following columns:
#
- `text_raw` – original subject + body
- `text_clean` – cleaned text used for modeling
- `label` – `0 = ham`, `1 = spam`
- `timestamp` – parsed send time (if available)
- `user_id` – recipient / mailbox owner (where available, e.g., TREC 2007)
- `source` – dataset name (`enron`, `spam_assassin`, `trec2007`)
#
Datasets used:
#
| Dataset          | Description                                          |
|------------------|------------------------------------------------------|
| **Enron**        | Corporate email dataset with ham/spam labels         |
| **SpamAssassin** | Classic spam corpus with clear spam features         |
| **TREC 2007**    | Research dataset including sender metadata           |
| **Combined**     | Union of the three datasets above                    |
#
To use your own dataset, you can place a CSV under `data/raw_data/` in the project root and extend the
loaders in `code/baselines.py` (and optionally the BERT / RoBERTa evaluation code).
#
---
#
## 📁 Project Structure
#
```text
.
├── code/
│   ├── baselines.py      # traditional ML pipeline (TF-IDF + NB/LogReg/SVM)
│   ├── bert_eval.py      # DistilBERT + LoRA experiments
│   └── device.py         # optional helper for selecting device (GPU/CPU)
│
├── traditional_output/   # results from traditional baselines
│   ├── random_split/     # RoBERTa confusion matrices for random splits
│   ├── time_split/       # RoBERTa confusion matrices for time-aware splits
│   ├── cross_dataset/    # RoBERTa confusion matrices for cross-dataset eval
│   ├── learning_curve/   # RoBERTa confusion matrices + training-size curves
│   └── summary/          # RoBERTa metrics CSVs (mirrors bert_output layout)
│
├── bert_output/          # results from DistilBERT + LoRA experiments
│   ├── random_split/     # confusion matrices for random splits
│   ├── time_split/       # confusion matrices for time-aware splits
│   ├── cross_dataset/    # confusion matrices for cross-dataset evaluation
│   ├── training_size/    # confusion matrices + training-size curves
│   ├── summary/          # metrics CSVs for DistilBERT experiments
│
├── roberta_output/       # results from RoBERTa experiments
│   ├── random_split/     # RoBERTa confusion matrices for random splits
│   ├── time_split/       # RoBERTa confusion matrices for time-aware splits
│   ├── cross_dataset/    # RoBERTa confusion matrices for cross-dataset eval
│   ├── learning_curve/   # RoBERTa confusion matrices + training-size curves
│   └── summary/          # RoBERTa metrics CSVs (mirrors bert_output layout)
│
├── requirements.txt            # Python dependencies
├── LICENSE
└── README.md
```
#
---
#
## ⚙️ 1. Traditional Baselines (`code/baselines.py`)
#
### 1.1 Preprocessing and Standardization
#
`baselines.py` implements the classical ML pipeline:
#
1. **Robust CSV loading**
   - Handles UTF-8 / Latin-1 encodings and skips malformed lines.
2. **Label normalization**
   - `norm_label` converts labels such as `"spam"`, `"ham"`, `1`, `0`, `"yes"`, `"no"` into clean `0/1`.
3. **Text cleaning**
   - HTML stripping with BeautifulSoup  
   - URL + email address removal  
   - Keep alphabetic tokens only, lowercase  
   - Tokenization + stopword removal using NLTK when available (with a fallback list otherwise)
4. **Standardization**
   - Outputs standardized DataFrames with `text_clean`, `label`, `timestamp`, `user_id`, `source`.  
   - Saves cleaned CSVs into `data/processed_data/`.  
   - Builds a `combined_clean.csv` by concatenating all standardized datasets.
#
Function responsible: `standardize_and_save()`.
#
### 1.2 Exploratory Data Analysis (EDA)
#
Function: `run_eda(data)`
#
- Computes per-dataset:
  - Number of samples
  - Spam rate
  - Average length of spam vs ham emails
- Saves:
  - `data/eda/eda_summary.csv`
  - `data/eda/class_proportion.png`
#
### 1.3 Baseline Models & Experiments
#
Function: `run_baselines(data)`
#
- TF–IDF vectorization:
  - `TfidfVectorizer(min_df=2, max_df=0.95, ngram_range=(1, 2))`
- Models:
  - Multinomial Naive Bayes
  - Logistic Regression (`class_weight="balanced"`)
  - LinearSVC (`class_weight="balanced"`)
#
**Experiments:**
#
1. **Random split (per dataset)**
   - Stratified 80/20 train–test split on:
     - Enron, SpamAssassin, TREC 2007, Combined
   - Confusion matrices:
     - `traditional_output/random_split/cm_<tag>_<model>.png`
   - Metrics:
     - `traditional_output/summary/baseline_metrics.csv`
     - `traditional_output/summary/baseline_metrics_pivot.csv`
#
2. **Time-aware split (per dataset)**
   - For datasets with timestamps (Enron, SpamAssassin):
     - Sort by timestamp.
     - Train on earliest 80%; test on latest 20%.
   - Confusion matrices:
     - `traditional_output/time_split/cm_<tag>_<model>.png`
#
### 1.4 Per-User Analysis
#
Function: `per_user_stats(df, min_samples=10)`
#
- Uses the TREC 2007 dataset (which includes `user_id`).  
- Trains a Multinomial Naive Bayes classifier and evaluates per user.  
- Writes precision, recall, and F1 for each user to:
  - `per_user_stats.csv`.
#
### 1.5 Running Traditional Baselines
#
```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows
#
pip install -r requirements.txt  # or install sklearn, nltk, bs4, matplotlib, etc.
#
python code/baselines.py
python code/extra_evals.py
```
#
This will:
#
- Generate cleaned datasets under `data/processed_data/`.  
- Produce EDA under `data/eda/`.  
- Run the traditional baselines and write all outputs into `traditional_output/`.
#
---
#
## 🤖 2. DistilBERT + LoRA Experiments (`code/bert_eval.py`)
#
DistilBERT experiments reuse the standardized CSVs and evaluate more powerful
semantic models with LoRA for parameter-efficient finetuning.
#
### 2.1 Key Details
#
- Model: `distilbert-base-uncased`
- Tokenizer: `DistilBertTokenizerFast` with `max_length=256`
- LoRA configuration:
  - `r = 8`, `lora_alpha = 16`, `lora_dropout = 0.1`, `bias = "none"`
- Training hyperparameters:
  - Batch size: 8
  - Epochs: 3
  - Learning rate: `2e-5`
  - Weight decay: `0.01`
  - Seed: 42
#
All runs log:
#
- `acc`, `prec`, `recall`, `f1`
- `n_train`, `n_test`
- `train_frac_pool` and `train_frac_total`
#
### 2.2 DistilBERT Experiments
#
1. **Baseline (Random Split)**
   - Stratified 80/20 split on each dataset (Enron, SpamAssassin, TREC 2007, Combined).
   - Confusion matrices:
     - `bert_output/random_split/cm_<tag>_DistilBERT_LoRA.png`
   - Metrics:
     - `bert_output/summary/distilbert_lora_random_split.csv`
#
2. **Time-Aware Split**
   - For Enron and SpamAssassin:
     - Sort by `timestamp`.
     - Train on earliest 80% and test on latest 20%.
   - Confusion matrices:
     - `bert_output/time_split/cm_<tag>_DistilBERT_LoRA.png`
   - Metrics:
     - `bert_output/summary/distilbert_lora_time_split.csv`
#
3. **Cross-Dataset Evaluation**
   - Train on one dataset (full), test on each of the other two:
     - e.g., `enron → spam_assassin`, `trec2007 → enron`, etc.
   - Confusion matrices:
     - `bert_output/cross_dataset/cm_cross_<train>_to_<test>_DistilBERT_LoRA.png`
   - Metrics:
     - `bert_output/summary/distilbert_lora_cross_dataset.csv`
#
4. **Training-Size Experiment (Combined)**
   - Fix **30% of the entire combined dataset** as a test set.  
   - From the remaining 70%, train on **10%, 20%, 30%, 40%, 50%, 60%, 70% of the entire dataset**.  
   - For each training size:
     - Train DistilBERT+LoRA on the sample.
     - Evaluate on the fixed 30% test set.
   - Outputs:
     - Confusion matrices:
       - `bert_output/training_size/cm_combined_trainsize_<XX>_DistilBERT_LoRA.png`
     - Metrics:
       - `bert_output/summary/distilbert_lora_training_size.csv`
     - Curves:
       - `bert_output/training_size/training_size_curves.png` (F1 & accuracy vs training fraction)
#
5. **Combined Metrics**
   - All runs (baseline, time, cross, training-size) are also aggregated into:
     - `bert_output/summary/distilbert_lora_all_experiments.csv`
#
### 2.3 Running DistilBERT Experiments
#
After running `code/baselines.py` once (to create `data/processed_data/*.csv`):
#
```bash
source .venv/bin/activate
#
pip install -r requirements.txt   # includes transformers, datasets, peft, etc.
#
python code/bert_eval.py
```
#
This will populate the `bert_output/` directory with all DistilBERT+LoRA metrics and plots.
#
---
#
## 🤖 3. RoBERTa Experiments 
#
RoBERTa experiments follow the **same data splits and evaluation protocols** as DistilBERT,
but with a different Transformer backbone.
#
- Results are stored under:
  - `roberta_output/random_split/`
  - `roberta_output/time_split/`
  - `roberta_output/cross_dataset/`
  - `roberta_output/learning_curve/`
  - `roberta_output/summary/`
#
Metrics and plots are organized in the same way as DistilBERT for easy
comparison.
#
---
#
## 📄 Metrics Source
#
**Traditional models (TF-IDF + NB/LogReg/SVM):**
#
- `traditional_output/summary/baseline_metrics.csv`
- `traditional_output/summary/baseline_metrics_pivot.csv`
- Confusion matrices:
  - `traditional_output/random_split/`
  - `traditional_output/time_split/`
#
**DistilBERT + LoRA:**
#
- `bert_output/summary/distilbert_lora_random_split.csv`
- `bert_output/summary/distilbert_lora_time_split.csv`
- `bert_output/summary/distilbert_lora_cross_dataset.csv`
- `bert_output/summary/distilbert_lora_training_size.csv`
- `bert_output/summary/distilbert_lora_all_experiments.csv`
- Confusion matrices and curves:
  - `bert_output/random_split/`, `bert_output/time_split/`
  - `bert_output/cross_dataset/`, `bert_output/training_size/`
#
**RoBERTa:**
#
- Metrics and plots under `roberta_output` and its subfolders
  (mirroring the DistilBERT layout).
- For details on how to run the experiments, see code/roberta/Readme.md
---
#
## ⭐ Acknowledgments
#
This project builds upon the work of several open-source projects and public datasets.
Special thanks to the maintainers and contributors of:
#
- [Hugging Face Transformers](https://huggingface.co/transformers/) – DistilBERT, RoBERTa, and the Trainer API.
- [Scikit-learn](https://scikit-learn.org/) – classical ML algorithms and evaluation tools.
- [NLTK](https://www.nltk.org/) – tokenization and stopword lists.
- [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/) – HTML parsing and text extraction.
- [Enron Email Dataset](https://www.cs.cmu.edu/~enron/) – benchmark corporate email corpus.
- [SpamAssassin Public Corpus](https://spamassassin.apache.org/publiccorpus/) – classic spam/ham dataset.
- [TREC 2007 Spam Track Dataset](https://trec.nist.gov/data/spam.html) – realistic spam filtering benchmark.
#
Gratitude also goes to the broader open-source community for providing tools, documentation,
and datasets that make NLP research accessible to everyone.





