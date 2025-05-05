# bert-fake-news-detection

Fake News Detection with Transformers
Overview
This project fine‑tunes two pre‑trained transformer models (BERT‑base‑uncased and RoBERTa‑base) to automatically detect fake news articles. We start from a publicly available Kaggle dataset of raw headlines and article bodies, perform cleaning and exploratory data analysis, then train and evaluate both models on the cleaned data. The entire pipeline—from raw CSVs through final metrics and visualizations—is organized in this repository.
Repository Structure
.
├── a1_True.csv                     # Raw “real” news articles (21,417 rows)
├── a2_Fake.csv                     # Raw “fake” news articles (23,502 rows)
├── cleaned_fake_news_dataset.csv   # Cleaned and merged dataset used for modeling
├── Data_cleaning_EDA.ipynb         # Jupyter notebook: cleaning, EDA, word clouds, distribution plots
├── train_models.py                 # Python script: fine‑tune BERT & RoBERTa, save models & figures
├── README.md                       # This file
└── models/                         # (Generated) saved model checkpoints & artifacts

Data Preparation & EDA
1. **Raw data**
- `a1_True.csv`: original real news articles  
- `a2_Fake.csv`: original fake news articles
- Download the above files from https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

2. **Cleaning & analysis**  
- Run `Data_cleaning_EDA.ipynb` to:  
  - Merge the two CSVs  
  - Remove duplicates and handle missing values  
  - Generate subject‐distribution bar charts and label‐proportion pie charts  
  - Produce word clouds for both classes  
- Output: `cleaned_fake_news_dataset.csv`
Model Training
The file `train_models.py` implements the fine‑tuning pipeline:
1. **Load** `cleaned_fake_news_dataset.csv`
2. **Split** into train (65%), validation (15%), and test (20%) with stratification
3. **Tokenize** texts with the Hugging‑Face tokenizer for each model
4. **Build** `tf.data.Dataset` objects and compile `TFBertForSequenceClassification` and `TFRobertaForSequenceClassification`
5. **Train** each model for 2 epochs (batch size 32, max length 64, learning rate 2e‑5)
6. **Evaluate** on validation and test sets, output accuracy, precision/recall/F₁ and confusion matrices
7. **Save**
   - Model weights and tokenizer in `models/bert-base-uncased/` and `models/roberta-base/`
   - Training plots (`train_val_history.png`) and confusion matrices (`confusion_matrix.png`)

How to Run

1. **Data cleaning & EDA**  
   ```bash
   jupyter nbconvert --to notebook --execute Data_cleaning_EDA.ipynb
   ```
2. **Model training**  
   ```bash
   python train_models.py
   ```
Results & Artifacts
After training, you will find under `models/` for each checkpoint:
- `tf_model.h5` (TensorFlow weights)
- `tokenizer.json` & related files
- `train_val_history.png`
- `confusion_matrix.png`
