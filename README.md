# 📧 Spam Email Detection Project
This project is the GitHub repository for CS410 Final Project (Group 52) at UIUC.
---
## 🧠 Overview
This repository provides a framework for spam email detection and experimentation. It includes:
- 📊 Data Standardization — unified preprocessing across major public datasets:
- Enron Email Dataset
- SpamAssassin
- TREC 2007
- ⚙️ Baseline Models — classic machine learning benchmarks using TF–IDF:
- Multinomial Naive Bayes
- Logistic Regression
- Linear SVM
- 🤖 Transformer-Based Classifier (DistilBERT + LoRA) — fine-tuned for advanced semantic
understanding of email content, with several evaluation protocols (random split, time-aware,
cross-dataset, and training-size experiments).
- 📈 Automated Reporting — generates detailed evaluation metrics in CSV format and visual
confusion matrices for each experiment.
For additional RoBERTa experiments and results, see the companion repo:
🔗 https://github.com/yueqiangwu/CS409_final_project
---
## 🧩 Tested Datasets
All datasets are standardized to a common schema:
- text_raw – original subject + body
- text_clean – cleaned text used for modeling
- label – 0 = ham, 1 = spam
- timestamp – parsed send time (if available)
- user_id – recipient (where available, e.g., TREC 2007)
- source – dataset name (enron, spam_assassin, trec2007)
| Dataset | Description |
|------------------|------------------------------------------------------|
| Enron | Corporate email dataset with ham/spam labels |
| SpamAssassin | Classic spam corpus with clear spam features |
| TREC 2007 | Research dataset including sender metadata |
| Combined | Union of the three datasets above |
Raw CSVs (original inputs) live in:
```text
data/raw_data/
spam_assassin.csv
enron_spam_data.csv (or enron_clean.csv)
trec_2007_data.csv
```
After preprocessing, cleaned versions live in:
```text
data/processed_data/
spam_assassin_clean.csv
enron_clean.csv
trec2007_clean.csv
combined_clean.csv
```
To use your own dataset, place a CSV under data/raw_data/ and extend or modify the loaders in
code/baselines.py (and optionally the BERT loader in code/bert_eval.py).
---
## 📁 Project Structure
```text
.
├── code/
│ ├── baselines.py # traditional ML pipeline (TF-IDF + NB/LogReg/SVM)
│ ├── bert_eval.py # DistilBERT + LoRA experiments
│ └── device.py # optional helper for selecting device (GPU/CPU)
│
├── data/
│ ├── raw_data/ # original CSVs (input)
│ ├── processed_data/ # cleaned CSVs (output from preprocessing)
│ └── eda/ # EDA plots and summaries
│
├── traditional_output/ # results from traditional baselines
│ ├── random_split/ # confusion matrices for random train/test splits
│ ├── time_split/ # confusion matrices for time-aware splits
│ └── summary/ # baseline_metrics.csv and pivoted metrics
│
├── bert_output/ # results from DistilBERT + LoRA experiments
│ ├── random_split/ # confusion matrices for random splits
│ ├── time_split/ # confusion matrices for time-aware splits
│ ├── cross_dataset/ # confusion matrices for cross-dataset evaluation
│ ├── training_size/ # confusion matrices + training-size curves
│ ├── summary/ # metrics CSVs for all BERT experiments
│ └── runs/ # HuggingFace Trainer artifacts (checkpoints, logs)
│
├── per_user_stats.csv # per-user performance summary (traditional NB baseline)
├── LICENSE
└── README.md
```
> Note: in an earlier version, metrics were written under data/experiments/summary/.
> In this project layout, traditional methods now log into traditional_output/, and
> DistilBERT-based methods log into bert_output/.
---
## ⚙️ 1. Traditional Baselines (code/baselines.py)
### 1.1 Preprocessing and Standardization
baselines.py performs the end-to-end pipeline for traditional models:
1. Robust CSV loading
- Handles multiple encodings (UTF-8 / Latin-1).
- Skips malformed lines.
2. Label normalization
- norm_label converts labels such as "spam", "ham", 1, 0, "yes", "no" into clean 0/1.
3. Text cleaning
- HTML stripping with BeautifulSoup.
- URL + email address removal.
- Keep alphabetic tokens only, lowercase.
- Tokenization + stopword removal using NLTK (punkt, stopwords) when available, with a
fallback stopword list if NLTK resources cannot be downloaded.
4. Standardization
- Each dataset is converted into a consistent schema with text_clean, label, timestamp,
user_id, and source.
- Cleaned CSVs are saved into data/processed_data/.
- A combined_clean.csv file is created by concatenating all datasets.
Function responsible: standardize_and_save().
### 1.2 Exploratory Data Analysis (EDA)
Function: run_eda(data)
- Computes:
- Number of samples per dataset
- Spam rate
- Average length of spam vs ham emails
- Writes summary CSV to:
- data/eda/eda_summary.csv
- Produces a class-proportion bar plot:
- data/eda/class_proportion.png
### 1.3 Baseline Models
Function: run_baselines(data)
For each dataset (spam_assassin, enron, trec2007, combined):
- Vectorization
- TfidfVectorizer(min_df=2, max_df=0.95, ngram_range=(1, 2))
- Models
- MultinomialNB
- LogisticRegression with class_weight="balanced"
- LinearSVC with class_weight="balanced"
#### 1.3.1 Random Split Experiment
- Stratified train/test split: 80% train / 20% test.
- For each model:
- Train on TF-IDF vectors from the train split.
- Predict on the test split.
- Compute accuracy, precision, recall, and F1.
- Outputs:
- Confusion matrices:
- traditional_output/random_split/cm_<tag>_<model>.png
- Metrics:
- traditional_output/summary/baseline_metrics.csv
- traditional_output/summary/baseline_metrics_pivot.csv
#### 1.3.2 Time-Aware Experiment
- For datasets with usable timestamps (e.g., Enron, SpamAssassin):
- Sort by timestamp.
- Use the earliest 80% of emails as train, latest 20% as test.
- Outputs:
- Confusion matrices:
- traditional_output/time_split/cm_<tag>_<model>.png
- Metrics are also included in baseline_metrics.csv with split="time".
### 1.4 Per-User Analysis
Function: per_user_stats(df, min_samples=10)
- Uses TREC 2007 (which includes recipient information) and trains a Multinomial Naive Bayes model.
- Evaluates performance for each user_id with at least min_samples emails in the test set.
- Computes precision, recall, and F1 per user and writes them to:
- per_user_stats.csv.
- This highlights how spam filter performance can vary across different users.
### 1.5 How to Run Traditional Baselines
From the project root:
```bash
python -m venv .venv
source .venv/bin/activate # Linux/macOS
# .venv\Scripts\activate # Windows
pip install numpy pandas scikit-learn nltk beautifulsoup4 matplotlib
python code/baselines.py
```
This will:
- Clean the datasets into data/processed_data/.
- Generate EDA artifacts in data/eda/.
- Run Naive Bayes, Logistic Regression, and Linear SVM baselines.
- Save all traditional metrics and confusion matrices into traditional_output/.
---
## 🤖 2. DistilBERT + LoRA Experiments (code/bert_eval.py)
bert_eval.py adds a modern deep learning baseline using DistilBERT with LoRA adapters for
efficient finetuning. It reuses the cleaned CSVs from data/processed_data/ and evaluates
multiple train/test protocols.
### 2.1 Dependencies
```bash
pip install torch transformers datasets peft scikit-learn pandas matplotlib
```
### 2.2 Path Handling
Paths in bert_eval.py are resolved relative to the repo:
```python
THIS_DIR = os.path.dirname(os.path.abspath(file)) # .../code
PROJECT_ROOT = os.path.dirname(THIS_DIR) # repo root
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
DIR_PROC = os.path.join(DATA_ROOT, "processed_data")
BERT_ROOT = os.path.join(PROJECT_ROOT, "bert_output")
DIR_RS = os.path.join(BERT_ROOT, "random_split")
DIR_TS = os.path.join(BERT_ROOT, "time_split")
DIR_CD = os.path.join(BERT_ROOT, "cross_dataset")
DIR_TRS = os.path.join(BERT_ROOT, "training_size")
DIR_SUM = os.path.join(BERT_ROOT, "summary")
DIR_RUNS = os.path.join(BERT_ROOT, "runs")
```
This matches the folder layout shown earlier and avoids hard-coded absolute paths.
### 2.3 Model & Tokenization
- Tokenizer: DistilBertTokenizerFast with max_length=256.
- Base model: DistilBertForSequenceClassification with num_labels=2.
- LoRA configuration (via peft.LoraConfig):
- task_type = TaskType.SEQ_CLS
- r = 8
- lora_alpha = 16
- lora_dropout = 0.1
- bias = "none"
- Training hyperparameters:
- Batch size: 8 (train & eval)
- Epochs: 3
- Learning rate: 2e-5
- Weight decay: 0.01
- Random seed: 42
Tokenization is applied through the HuggingFace datasets library with
padding/truncation to a fixed length, and the resulting Datasets are passed to
transformers.Trainer.
### 2.4 Shared Metrics Schema
Every DistilBERT run returns a metrics row with:
- model – "DistilBERT_LoRA"
- exp_group – "baseline", "time", "cross", "train_size"
- split – "random", "time", "cross", "train_size"
- tag – label for the run (e.g. combined_rand, cross_enron_to_trec2007)
- train_dataset, test_dataset
- n_train, n_test
- train_frac_pool – fraction of the available train pool used
- train_frac_total – fraction of the entire source dataset used for training
- acc, prec, recall, f1
Confusion matrices are generated to mirror the style of the traditional baselines:
- Rows: true class (Ham(0), Spam(1))
- Columns: predicted class (Ham(0), Spam(1))
### 2.5 DistilBERT Experiments
#### 2.5.1 Baseline – Random Split
Function: run_baseline_experiments(data_dict)
- Datasets: spam_assassin, enron, trec2007, combined.
- Stratified 80/20 train/test split on each dataset.
- Train DistilBERT+LoRA on train, evaluate on test.
Outputs:
- Confusion matrices:
- bert_output/random_split/cm_<tag>_DistilBERT_LoRA.png
- Metrics:
- bert_output/summary/distilbert_lora_random_split.csv
#### 2.5.2 Time-Aware Split
Function: run_time_aware_experiments(data_dict)
- Datasets with timestamps: currently spam_assassin and enron.
- Sort by timestamp.
- Earliest 80% → train, latest 20% → test.
- Evaluates how performance holds up under a more realistic temporal split.
Outputs:
- Confusion matrices:
- bert_output/time_split/cm_<tag>_DistilBERT_LoRA.png
- Metrics:
- bert_output/summary/distilbert_lora_time_split.csv
#### 2.5.3 Cross-Dataset Evaluation
Function: run_cross_dataset_experiments(data_dict)
- Train on one dataset, test on the other two:
- spam_assassin → enron, spam_assassin → trec2007
- enron → spam_assassin, enron → trec2007
- trec2007 → spam_assassin, trec2007 → enron
- Uses the full source dataset as training data.
Outputs:
- Confusion matrices:
- bert_output/cross_dataset/cm_cross_<train>_to_<test>_DistilBERT_LoRA.png
- Metrics:
- bert_output/summary/distilbert_lora_cross_dataset.csv
This highlights how well a model trained on one corpus generalizes to another.
#### 2.5.4 Training-Size Experiment (Combined Dataset)
Function: run_training_size_experiments(combined_df)
Objective: determine how much training data is needed for DistilBERT+LoRA to reach near-max
performance.
- Fix 30% of the entire combined dataset as a test set (stratified).
- The remaining 70% is the train pool.
- For training, use increasing fractions of the entire dataset:
- 10%, 20%, 30%, 40%, 50%, 60%, 70%
- Each value corresponds to a specific number of training samples drawn from the 70% pool.
- For each training size:
- Train DistilBERT+LoRA on the sampled training subset.
- Evaluate on the same fixed 30% test set.
Outputs:
- Confusion matrices:
- bert_output/training_size/cm_combined_trainsize_<XX>_DistilBERT_LoRA.png
where <XX> is the percentage of the entire dataset used for training.
- Metrics:
- bert_output/summary/distilbert_lora_training_size.csv
- Convenience plot:
- bert_output/training_size/training_size_curves.png
- Left panel: F1 vs training percentage
- Right panel: Accuracy vs training percentage
These curves allow visual inspection of where marginal gains flatten out, giving an empirical
“upper bound” on how much labeled data is needed.
### 2.6 Combined Metrics
After running all experiments, bert_eval.py also writes:
- bert_output/summary/distilbert_lora_all_experiments.csv
This collects every DistilBERT+LoRA run (baseline, time-aware, cross-dataset, training-size) in a
single table for downstream analysis or plotting.
### 2.7 How to Run DistilBERT Experiments
From the project root (after running the traditional baselines at least once to create
data/processed_data/*.csv):
```bash
source .venv/bin/activate # or any other environment
pip install torch transformers datasets peft scikit-learn pandas matplotlib
python code/bert_eval.py
```
This will:
- Load the standardized CSVs from data/processed_data/.
- Run all DistilBERT+LoRA experiments (random split, time-aware, cross-dataset, training-size).
- Store all metrics and plots under bert_output/.
---
## 📄 Metrics Source
Traditional baselines (TF-IDF + NB/LogReg/SVM):
- traditional_output/summary/baseline_metrics.csv
- traditional_output/summary/baseline_metrics_pivot.csv
- Confusion matrices under:
- traditional_output/random_split/
- traditional_output/time_split/
DistilBERT + LoRA experiments:
- bert_output/summary/distilbert_lora_random_split.csv
- bert_output/summary/distilbert_lora_time_split.csv
- bert_output/summary/distilbert_lora_cross_dataset.csv
- bert_output/summary/distilbert_lora_training_size.csv
- Aggregated:
- bert_output/summary/distilbert_lora_all_experiments.csv
- Confusion matrices and training-size curves under the respective folders in bert_output/.
Earlier iterations also wrote BERT metrics into
data/experiments/summary/baseline_metrics_bert.csv;
the current layout uses bert_output/summary/* instead.
---
## ⭐ Acknowledgments
This project builds upon the work of several open-source projects and public datasets.
Special thanks to the maintainers and contributors of the following:
- Hugging Face Transformers
 – for providing
state-of-the-art NLP models such as DistilBERT and RoBERTa.
- Scikit-learn
 – for classical machine learning algorithms and
evaluation tools.
- NLTK (Natural Language Toolkit)
 – for text preprocessing,
tokenization, and stopword support.
- BeautifulSoup4
 – for HTML text extraction
and cleaning.
- Enron Email Dataset
 – a benchmark dataset of corporate
emails.
- SpamAssassin Public Corpus
 – one of the
most widely used spam/ham datasets.
- TREC 2007 Spam Track Dataset
 – for research on
realistic email filtering.
Gratitude also goes to the open-source community for providing tools, documentation, and datasets
that make NLP research accessible to everyone.
