import pandas as pd
import pytest
from src.data_processing import clean_text, apply_cleaning, tokenize_text, apply_tokenization, split_data
from transformers import AutoTokenizer

def test_clean_text():
    # Test case 1: Basic cleaning
    text = "This is a TeXt with  some   SPACES and 123! #@$"
    expected = "this is a text with some spaces and 123"  # Removed space at the end
    assert clean_text(text) == expected

def test_apply_cleaning():
    # Test case 1: Basic DataFrame cleaning
    data = {'content': ["This is a TeXt with  some   SPACES and 123! #@$", "Another Example"]}
    example_df = pd.DataFrame(data)
    
    cleaned_df = apply_cleaning(example_df)
    
    expected_data = {'content': ["This is a TeXt with  some   SPACES and 123! #@$", "Another Example"],
                     'cleaned_content': ["this is a text with some spaces and 123", "another example"]}  # Removed space at the end
    expected_df = pd.DataFrame(expected_data)

    pd.testing.assert_frame_equal(cleaned_df, expected_df)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def test_tokenize_text():
    # Test case 1: Basic tokenization
    text = "This is a test sentence."
    tokens = tokenize_text(text, tokenizer)
    assert tokens['input_ids'].shape == (1, 128)  # Check input_ids shape
    assert tokens['attention_mask'].shape == (1, 128)  # Check attention_mask shape

def test_apply_tokenization():
    # Test case 1: Basic DataFrame tokenization
    data = {'cleaned_content': ["This is a test sentence.", "Another sentence."]}
    example_df = pd.DataFrame(data)
    tokenized_df = apply_tokenization(example_df, tokenizer)
    assert 'tokens' in tokenized_df.columns
    assert tokenized_df['tokens'][0]['input_ids'].shape == (1, 128)

def test_split_data():
    # Test case 1: Basic data splitting
    data = {'cleaned_content': ["This is a test sentence.", "Another sentence.", "Yet another sentence", "A fourth sentence."]}
    example_df = pd.DataFrame(data)
    train_df, val_df = split_data(example_df, test_size=0.25, random_state=42)
    assert len(train_df) == 3
    assert len(val_df) == 1
