import pytest
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.inference import predict_sentiment


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("bert-base-uncased")


@pytest.fixture
def dummy_model():
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    return model


def test_predict_sentiment_positive(dummy_model, tokenizer):
    text = "I love this! It's amazing and very well made."
    sentiment = predict_sentiment(text, dummy_model, tokenizer)
    assert sentiment in ["positive", "negative"]  # Can't force dummy model, just test shape/robustness


def test_predict_sentiment_negative(dummy_model, tokenizer):
    text = "I hate this. It's the worst purchase I've made."
    sentiment = predict_sentiment(text, dummy_model, tokenizer)
    assert sentiment in ["positive", "negative"]


def test_predict_sentiment_neutral(dummy_model, tokenizer):
    text = "This product is okay, not good, not bad."
    sentiment = predict_sentiment(text, dummy_model, tokenizer)
    assert sentiment in ["positive", "negative"]
