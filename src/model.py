import os
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score
from src.data_processing import apply_cleaning, apply_tokenization, split_data
from src.data_extraction import load_data


def configure_tokenizer(model_name="bert-base-uncased", max_len=128):
    """Helper function to configure the tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def configure_model(model_name="bert-base-uncased", num_labels=2):
    """Helper function to configure the model."""
    config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    return model


class SentimentDataset(torch.utils.data.Dataset):
    """Custom dataset for sentiment classification."""
    
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


def train_model(config, train_dataloader, val_dataloader, epochs=3, device="cpu"):
    """Train the BERT model on provided data."""
    model = config['model']
    optimizer = config['optimizer']
    scheduler = config['scheduler']
    
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
        print(f"Epoch {epoch + 1}: Training Loss = {avg_loss:.4f}")
        evaluate_model(model, val_dataloader, device)


def evaluate_model(model, val_dataloader, device="cpu"):
    """Evaluate the model on a validation set."""
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
    """Save the trained model."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


def load_trained_model(model_name="bert-base-uncased", filepath="saved_models/bert_sentiment.pth"):
    """Load a pre-trained BERT model with trained weights."""
    config = AutoConfig.from_pretrained(model_name, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        ignore_mismatched_sizes=True
    )
    model.load_state_dict(torch.load(filepath, map_location=torch.device('cpu')))
    return model


def prepare_data(file_path='dataset.csv', tokenizer=None):
    """Prepare the dataset for training and validation."""
    df = load_data(file_path)

    cleaned_df = apply_cleaning(df)
    tokenized_df = apply_tokenization(cleaned_df, tokenizer)
    train_df, val_df = split_data(tokenized_df)

    train_dataset = SentimentDataset(train_df, tokenizer)
    val_dataset = SentimentDataset(val_df, tokenizer)

    return train_dataset, val_dataset


if __name__ == '__main__':
    FILE_PATH = 'dataset.csv'

    tokenizer = configure_tokenizer()
    train_dataset, val_dataset = prepare_data(FILE_PATH, tokenizer)

    BATCH_SIZE = 32
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = configure_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * 3
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    config = {
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler
    }

    train_model(config, train_loader, val_loader, epochs=1, device=device)
    save_model(model, "saved_models/bert_sentiment.pth")
