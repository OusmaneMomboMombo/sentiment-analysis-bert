import pytest
import torch
from transformers import AutoTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
from src.model import (
    load_model, train_model, evaluate_model,
    SentimentDataset, load_trained_model, save_model
)


@pytest.fixture
def model():
    return load_model(num_labels=2)


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("bert-base-uncased")


@pytest.fixture
def sample_dataframe():
    import pandas as pd
    data = {
        'content': ["This is a positive review", "This is a negative review"],
        'score': [5, 1]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_dataset(sample_dataframe, tokenizer):
    return SentimentDataset(sample_dataframe, tokenizer)


@pytest.fixture
def sample_dataloader(sample_dataset):
    return DataLoader(sample_dataset, batch_size=2)


def test_load_model(model):
    # ✅ Correction : on vérifie contre le vrai type retourné
    assert isinstance(model, BertForSequenceClassification)


def test_model_output_shape(model, sample_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch = next(iter(sample_dataloader))

    with torch.no_grad():
        outputs = model(batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device))

    assert outputs.logits.shape == (2, 2)


def test_train_model(model, sample_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    try:
        train_model(model, sample_dataloader, sample_dataloader, optimizer, scheduler, epochs=1, device=device)
    except Exception as e:
        pytest.fail(f"Training failed: {e}")


def test_evaluate_model(model, sample_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc = evaluate_model(model, sample_dataloader, device=device)
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


def test_save_load_model(model, tmp_path):
    path = tmp_path / "model.pth"
    save_model(model, str(path))
    assert path.exists()

    loaded_model = load_trained_model(filepath=str(path))
    assert isinstance(loaded_model, BertForSequenceClassification)
