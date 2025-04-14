"""
Module for sentiment prediction using a trained model and tokenizer.
Provides a CLI interface for prediction.
"""

import os
import sys
import torch
from transformers import AutoTokenizer
from src.model import load_trained_model
from src.data_processing import clean_text

# Ajout du dossier parent au path (utile pour exécution directe)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def predict_sentiment(input_text, model, tokenizer, device="cpu"):
    """
    Predicts the sentiment of a given text.
    
    Args:
        input_text (str): The text to analyze.
        model: The trained sentiment classification model.
        tokenizer: The tokenizer corresponding to the model.
        device (str): Device to run the inference on ("cpu" or "cuda").
    
    Returns:
        str: "positive" or "negative" based on the model's prediction.
    """
    model.eval()
    model.to(device)

    cleaned_text = clean_text(input_text)

    encoded = tokenizer.encode_plus(
        cleaned_text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    return "positive" if pred == 1 else "negative"


if __name__ == '__main__':
    MODEL_PATH = "/app/saved_models/bert_sentiment.pth"
    model_instance = load_trained_model(filepath=MODEL_PATH)
    tokenizer_instance = AutoTokenizer.from_pretrained("bert-base-uncased")
    device_name = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Mode CLI
    if len(sys.argv) > 1:
        input_text = " ".join(sys.argv[1:])
        result_sentiment = predict_sentiment(input_text, model_instance, tokenizer_instance, device_name)
        print(f"👉 Sentiment: {result_sentiment.upper()}")
    else:
        print("Usage: docker run -v /chemin/saved_models:/app/saved_models sentiment-cli 'Votre texte ici'")
