import json
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, get_scheduler
from datasets import Dataset
from tqdm.auto import tqdm

def train_sklearn_pipeline(train_path: str, test_path: str, text_col: str, label_col: str) -> str:
    """Train a Scikit-Learn TF-IDF + LogisticRegression pipeline."""
    print(f"Loading datasets from {train_path} and {test_path}...")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    X_train = df_train[text_col].fillna("")
    y_train = df_train[label_col]
    X_test = df_test[text_col].fillna("")
    y_test = df_test[label_col]
    
    # TODO: Initialize a Scikit-Learn Pipeline with TfidfVectorizer (named 'tfidf') and LogisticRegression (named 'clf')
    pipeline = None 
    
    # Uncomment the following lines after initializing the pipeline
    """
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()
    y_test_labels = [str(l) for l in sorted(list(set(y_test)))]
    
    return json.dumps({
        "status": "success",
        "model": "sklearn_logistic_regression",
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm,
        "labels": y_test_labels
    })
    """
    return json.dumps({"status": "TODO"})

def train_bert_pipeline(train_path: str, test_path: str, text_col: str, label_col: str, epochs: int) -> str:
    """Fine-tune a DistilBERT model."""
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    df_train['text'] = df_train[text_col].astype(str)
    df_test['text'] = df_test[text_col].astype(str)
    
    labels = sorted(df_train[label_col].unique())
    label2id = {label: i for i, label in enumerate(labels)}
    df_train['labels'] = df_train[label_col].map(label2id)
    df_test['labels'] = df_test[label_col].map(label2id)

    train_dataset = Dataset.from_pandas(df_train)
    test_dataset = Dataset.from_pandas(df_test)
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    
    def tokenize_function(examples):
        # TODO: call tokenizer on examples['text'], pass truncation=True, max_length=128, and padding='max_length'
        pass

    # Uncomment the following snippet after completing the tokenize_function TODO
    """
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)
    
    tokenized_train.set_format("torch", columns=['input_ids', 'attention_mask', 'labels'])
    tokenized_test.set_format("torch", columns=['input_ids', 'attention_mask', 'labels'])

    # TODO: Create a DataLoader for `tokenized_train` (shuffle=True) and `tokenized_test`. 
    # Use batch_size=4.
    train_dataloader = None
    eval_dataloader = None
    
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(labels))
    
    # TODO: Initialize AdamW optimizer with lr=5e-5 and the linear scheduler.
    optimizer = None
    lr_scheduler = None
    
    device = torch.device("cpu")
    model.to(device)
    
    # TODO: Implement the training loop for the given number of epochs.
    # Inside the loop: batch to device, model forward, loss backward, optimizer step, scheduler step, zero grad.
    """
    model.train()
    for epoch in range(epochs):
        for batch in train_dataloader:
            # batch = {k: v.to(device) for k, v in batch.items()}
            # ...
            pass

    
    # Evaluation loop
    model.eval()
    y_pred, y_test = [], []
    # TODO: Iterate over eval_dataloader to get predictions
    
    """
    acc = accuracy_score(y_test, y_pred)
    string_labels = [str(l) for l in labels]
    report = classification_report(y_test, y_pred, target_names=string_labels, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    return json.dumps({
        "status": "success",
        "model": "distilbert",
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm,
        "labels": string_labels
    })
    """
    return json.dumps({"status": "TODO"})



def plot_confusion_matrix(matrix_data: list, labels: list) -> str:
    """Plot a confusion matrix and save it to disk."""
    plt.figure(figsize=(8, 6))
    
    # TODO: Using seaborn (sns), plot a heatmap of `matrix_data`. Add `annot=True`, `fmt='d'`, and `cmap='Blues'`. 
    # Provide `labels` for both `xticklabels` and `yticklabels` parameters.
    
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    os.makedirs('data/outputs', exist_ok=True)
    save_path = 'data/outputs/confusion_matrix.png'
    
    # TODO: Save the matplotlib plot to `save_path` using plt.savefig()
    
    # plt.close()
    
    return json.dumps({"status": "success", "message": f"Plot saved to {save_path}"})
