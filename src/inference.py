import torch
from transformers import AutoTokenizer
from src.model import load_trained_model
from src.data_processing import clean_text


def predict_sentiment(text, model, tokenizer, device="cpu"):
    """
    Predicts the sentiment of a given text.
    Returns "positive" or "negative".
    """
    model.eval()
    model.to(device)

    cleaned_text = clean_text(text)

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
    model_path = "saved_models/bert_sentiment.pth"
    model = load_trained_model(filepath=model_path)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    texts = [
        "This is a great product! I highly recommend it.",
        "This is a terrible product! I do not recommend it."
    ]

    for text in texts:
        sentiment = predict_sentiment(text, model, tokenizer, device)
        print(f"Text: \"{text}\" → Sentiment: {sentiment}")
