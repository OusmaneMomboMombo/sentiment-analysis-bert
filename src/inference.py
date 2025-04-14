"""
Module for sentiment prediction using a trained model and tokenizer.
Provides a CLI interface for prediction.
"""

import sys
import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Ajout du dossier parent au path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def predict_sentiment(text, model, tokenizer, device="cpu"):
    """Prédiction du sentiment"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits).item()
    
    return "positive" if pred == 1 else "negative"

if __name__ == "__main__":
    MODEL_DIR = "./saved_models"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # Chargement
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
        
        # Interface CLI
        if len(sys.argv) > 1:
            text = " ".join(sys.argv[1:])
            print(f"👉 Sentiment: {predict_sentiment(text, model, tokenizer, device).upper()}")
        else:
            print("Usage: python inference.py 'Votre texte'")
            
    except Exception as e:
        print(f"❌ Erreur: {e}")
        print("Vérifie que le dossier saved_models contient bien tous les fichiers nécessaires")