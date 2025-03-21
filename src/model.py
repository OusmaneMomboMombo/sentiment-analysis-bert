import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    AdamW,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score
import os
import pandas as pd

from src.data_processing import clean_text, apply_cleaning, apply_tokenization, split_data
from src.data_extraction import load_data


# --- Classe SentimentDataset ---
class SentimentDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=128):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        text = self.dataframe['content'].iloc[index]
        score = self.dataframe['score'].iloc[index]
        label = 1 if score >= 3 else 0

        tokens = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# --- Fonctions modèle ---
def load_model(model_name="bert-base-uncased", num_labels=2):
    config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    return model


def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, epochs=3, device="cpu"):
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}: Training Loss = {avg_loss:.4f}")
        evaluate_model(model, val_dataloader, device)


def evaluate_model(model, val_dataloader, device="cpu"):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Validation Accuracy: {accuracy:.4f}")
    return accuracy


def save_model(model, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


def load_trained_model(model_name="bert-base-uncased", num_labels=2, filepath="model.pth"):
    config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    model.load_state_dict(torch.load(filepath))
    return model


# --- Main ---
if __name__ == '__main__':
    file_path = 'dataset.csv'
    df = load_data(file_path)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    cleaned_df = apply_cleaning(df)
    tokenized_df = apply_tokenization(cleaned_df, tokenizer)
    train_df, val_df = split_data(tokenized_df)

    train_dataset = SentimentDataset(train_df, tokenizer)
    val_dataset = SentimentDataset(val_df, tokenizer)

    BATCH_SIZE = 32
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = load_model(num_labels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * 3
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    train_model(model, train_loader, val_loader, optimizer, scheduler, epochs=1, device=device)
    save_model(model, "saved_models/bert_sentiment.pth")
