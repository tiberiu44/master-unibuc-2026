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
    """
    Train a Scikit-Learn TF-IDF + LogisticRegression pipeline.
    """
    print(f"Loading datasets from {train_path} and {test_path}...")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    X_train = df_train[text_col].fillna("")
    y_train = df_train[label_col]
    X_test = df_test[text_col].fillna("")
    y_test = df_test[label_col]
    
    print("Initializing TF-IDF + LogisticRegression pipeline...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression())
    ])
    
    print("Fitting model (this may take a moment)...")
    pipeline.fit(X_train, y_train)
    
    print("Generating predictions and metrics...")
    y_pred = pipeline.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    # Need string labels for JSON
    y_test_labels = [str(l) for l in sorted(list(set(y_test)))]
    
    return json.dumps({
        "status": "success",
        "model": "sklearn_logistic_regression",
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm,
        "labels": y_test_labels
    })

def train_bert_pipeline(train_path: str, test_path: str, text_col: str, label_col: str, epochs: int) -> str:
    """
    Fine-tune a DistilBERT model.
    """
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Using small subsets to make training fast
    df_train = df_train.sample(min(1000, len(df_train))) 

    # Make sure text is string type
    df_train['text'] = df_train[text_col].astype(str)
    df_test['text'] = df_test[text_col].astype(str)
    
    # Map labels to ints
    labels = sorted(df_train[label_col].unique())
    label2id = {label: i for i, label in enumerate(labels)}
    df_train['labels'] = df_train[label_col].map(label2id)
    df_test['labels'] = df_test[label_col].map(label2id)

    train_dataset = Dataset.from_pandas(df_train)
    test_dataset = Dataset.from_pandas(df_test)
    
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=128, padding='max_length')

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    tokenized_train.set_format("torch", columns=['input_ids', 'attention_mask', 'labels'])
    tokenized_test.set_format("torch", columns=['input_ids', 'attention_mask', 'labels'])
    
    train_dataloader = DataLoader(tokenized_train, shuffle=True, batch_size=4)
    eval_dataloader = DataLoader(tokenized_test, batch_size=4)
    
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(labels))
    
    # Manual loop
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_training_steps = epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    
    device = torch.device("cpu") # Lab environment uses CPU
    model.to(device)
    
    print(f"Starting training for {epochs} epochs...")
    progress_bar = tqdm(range(num_training_steps))
    
    model.train()
    for epoch in range(epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            
    # Evaluation
    model.eval()
    y_pred = []
    y_test = []
    
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        y_pred.extend(predictions.cpu().numpy())
        y_test.extend(batch["labels"].cpu().numpy())
    
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

def plot_confusion_matrix(matrix_data: list, labels: list) -> str:
    """
    Plot a confusion matrix and save it to disk.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix_data, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    os.makedirs('data/outputs', exist_ok=True)
    save_path = 'data/outputs/confusion_matrix.png'
    plt.savefig(save_path)
    plt.close()
    
    return json.dumps({"status": "success", "message": f"Plot saved to {save_path}"})
