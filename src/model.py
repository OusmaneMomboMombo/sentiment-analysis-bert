import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    get_linear_schedule_with_warmup,
    BertForSequenceClassification,
    BertTokenizer
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.data_processing import clean_text, apply_cleaning, apply_tokenization, split_data
from src.data_extraction import load_data

class SentimentDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=128):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        text = str(self.dataframe['content'].iloc[index])
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

def load_model(model_name="bert-base-uncased", num_labels=2):
    """Charge le modèle pré-entraîné avec une nouvelle tête de classification"""
    try:
        # Essayer de charger depuis le dossier local d'abord
        config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=True
        )
        print(f"Modèle chargé depuis {model_name}")
        return model
    except Exception as e:
        print(f"Erreur de chargement: {e}")
        raise

def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, epochs=3, device="cpu"):
    """Entraîne le modèle avec évaluation périodique"""
    model.to(device)
    best_accuracy = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(train_dataloader)
        print(f"\nEpoch {epoch+1}:")
        print(f"Training Loss = {avg_loss:.4f}")
        
        # Évaluation
        val_accuracy, _ = evaluate_model(model, val_dataloader, device)
        
        # Sauvegarde du meilleur modèle
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            save_model(model, "saved_models/best_model")
            print(f"Nouveau meilleur modèle sauvegardé (Accuracy: {best_accuracy:.2%})")

def evaluate_model(model, dataloader, device="cpu", return_report=False):
    """Évalue le modèle et retourne les métriques"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Évaluation"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    
    if return_report:
        report = classification_report(all_labels, all_preds, target_names=['negative', 'positive'])
        print("\nRapport de classification:")
        print(report)
        return accuracy, report
    
    return accuracy

def load_trained_model(model_dir="saved_models"):
    """Charge un modèle entraîné depuis le dossier"""
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            local_files_only=True
        )
        print(f"Modèle chargé depuis {model_dir}")
        return model
    except Exception as e:
        print(f"Erreur de chargement: {e}")
        raise

def save_model(model, output_dir="saved_models"):
    """Sauvegarde le modèle complet avec tokenizer et config"""
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    print(f"Modèle complet sauvegardé dans {output_dir}")

if __name__ == '__main__':
    # Configuration
    DATA_PATH = 'dataset.csv'
    MODEL_NAME = "bert-base-uncased"
    EPOCHS = 3
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    
    # Préparation des données
    print("Chargement des données...")
    df = load_data(DATA_PATH)
    cleaned_df = apply_cleaning(df)
    
    print("Préparation des datasets...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_df, val_df = split_data(cleaned_df)
    
    train_dataset = SentimentDataset(train_df, tokenizer)
    val_dataset = SentimentDataset(val_df, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Initialisation du modèle
    print("Initialisation du modèle...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(MODEL_NAME)
    
    # Optimiseur et scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Entraînement
    print("Début de l'entraînement...")
    train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        epochs=EPOCHS,
        device=device
    )
    
    # Sauvegarde finale
    save_model(model)
    print("Entraînement terminé et modèle sauvegardé.")