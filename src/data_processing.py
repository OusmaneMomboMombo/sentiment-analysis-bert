"""
Module for text cleaning, tokenization, and dataset splitting.
"""

import re
from sklearn.model_selection import train_test_split

def clean_text(text):
    """
    Cleans the input text by removing special characters, converting to lowercase,
    and normalizing whitespace.
    """
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def apply_cleaning(df, text_column="content"):
    """
    Apply the `clean_text` function to the specified text column of a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the text data.
        text_column (str, optional): The name of the column containing the text.
            Defaults to "content".

    Returns:
        pd.DataFrame: A new DataFrame with the cleaned text column.
    """
    df = df.copy()
    df["cleaned_content"] = df[text_column].apply(clean_text)
    return df

def tokenize_text(text, tokenizer):
    """
    Tokenizes the input text using the specified tokenizer.

    Args:
        text (str): The input text to tokenize.
        tokenizer: A tokenizer instance (e.g., from Hugging Face transformers).

    Returns:
        dict: A dictionary of tokenized input, including input_ids and attention_mask.
    """
    return tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

def apply_tokenization(df, tokenizer, text_column="cleaned_content"):
    """
    Apply the `tokenize_text` function to the specified text column of a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the text.
        tokenizer: Tokenizer to use.
        text_column (str): Name of the column with text to tokenize.

    Returns:
        pd.DataFrame: DataFrame with a new 'tokens' column containing tokenized results.
    """
    df = df.copy()
    df['tokens'] = df[text_column].apply(lambda x: tokenize_text(x, tokenizer))
    return df

def split_data(df, test_size=0.2, random_state=42):
    """
    Splits the DataFrame into training and validation sets.

    Args:
        df (pd.DataFrame): The full dataset.
        test_size (float): Proportion of data to use for validation.
        random_state (int): Random seed for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Training and validation DataFrames.
    """
    return train_test_split(df, test_size=test_size, random_state=random_state)
