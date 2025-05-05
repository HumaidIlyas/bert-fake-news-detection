#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.python.platform import build_info as tf_build


# ——— Configuration ——————————————————————————————————————————————
CSV_PATH   = '/N/u/hilyas/BigRed200/Fake_News_Detection/cleaned_fake_news_dataset.csv'
OUTPUT_DIR = '/N/u/hilyas/BigRed200/Fake_News_Detection/models'
BATCH_SIZE = 32
EPOCHS     = 2
MAX_LEN    = 64

# List of 🤗 model checkpoints to compare
MODEL_NAMES = [
    'bert-base-uncased',
    'roberta-base',
]

# ——— Load & preprocess once ——————————————————————————————————————
df = pd.read_csv(CSV_PATH)
df = df.drop_duplicates(subset='news')  # guard against duplicates

# categorical label encoding
df['label'] = df['label'].astype('category').cat.codes
num_labels = df['label'].nunique()

# train/val/test split
tv_texts, test_texts, tv_lbls, test_lbls = train_test_split(
    df['news'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)
val_ratio = 0.15 / 0.8
train_texts, val_texts, train_lbls, val_lbls = train_test_split(
    tv_texts, tv_lbls, test_size=val_ratio, random_state=42, stratify=tv_lbls
)

# convert to numpy arrays
train_lbls = train_lbls.values.astype(np.int32)
val_lbls   = val_lbls.values.astype(np.int32)
test_lbls  = test_lbls.values.astype(np.int32)

# ensure text dtype
train_texts = train_texts.fillna('').astype(str)
val_texts   = val_texts.fillna('').astype(str)
test_texts  = test_texts.fillna('').astype(str)

# helper to tokenize and build tf.data.Dataset
def build_dataset(tokenizer, texts, labels, shuffle=False):
    enc = tokenizer(
        texts.tolist(),
        padding='max_length',
        truncation=True,
        max_length=MAX_LEN,
        return_tensors='tf'
    )
    ds = tf.data.Dataset.from_tensor_slices((
        {
            'input_ids':      enc['input_ids'],
            'attention_mask': enc['attention_mask'],
            'token_type_ids': enc.get('token_type_ids')  # some models omit
        },
        labels
    ))
    if shuffle:
        ds = ds.shuffle(1000)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# helper to plot history
def plot_history(history, out_file):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # loss
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    # accuracy
    axes[1].plot(history.history['accuracy'], label='Train Acc')
    axes[1].plot(history.history['val_accuracy'], label='Val Acc')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)

# helper to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, labels, out_file, model_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(f'{model_name} Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)

# ——— Loop over models —————————————————————————————————————————————
for model_name in MODEL_NAMES:
    print(f"\n\n======== TRAINING {model_name} ========")

    # setup save directory
    save_path = os.path.join(OUTPUT_DIR, model_name.replace('/', '_'))
    os.makedirs(save_path, exist_ok=True)

    # 1) Tokenizer & datasets
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_ds = build_dataset(tokenizer, train_texts, train_lbls, shuffle=True)
    val_ds   = build_dataset(tokenizer, val_texts,   val_lbls)
    test_ds  = build_dataset(tokenizer, test_texts,  test_lbls)

    # 2) Model init & compile
    model = TFAutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(2e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # 3) Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose=1
    )

    # 4) Metrics
    val_acc = max(history.history['val_accuracy'])
    print(f"{model_name} best val_acc = {val_acc:.4f}")

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"{model_name} test_acc = {test_acc:.4f}")

    # classification report
    y_probs = model.predict(test_ds).logits
    y_pred  = np.argmax(y_probs, axis=1)
    print(classification_report(test_lbls, y_pred, digits=4))

    # 5) Save model & tokenizer
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    # 6) Plot & save
    history_file = os.path.join(save_path, 'train_val_history.png')
    plot_history(history, history_file)
    print(f"Saved training curves to {history_file}")

    cm_labels = df['label'].astype('category').cat.categories.tolist()
    cm_file = os.path.join(save_path, 'confusion_matrix.png')
    plot_confusion_matrix(test_lbls, y_pred, cm_labels, cm_file, model_name)
    print(f"Saved confusion matrix to {cm_file}")

    print(f"Saved {model_name} artifacts to {save_path}")
