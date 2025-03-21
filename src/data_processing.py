import pandas as pd
import re
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

def clean_text(text):
    # Remove unnecessary characters (e.g., special characters, emojis)
    text = re.sub(r'[^\w\s]', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Normalize whitespace (remove multiple spaces, leading/trailing spaces)
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
    df = df.copy()  # Avoid modifying the original DataFrame
    df["cleaned_content"] = df[text_column].apply(clean_text)
    return df

def tokenize_text(text, tokenizer):
    """
    Tokenizes the input text using the specified tokenizer.
    """
    tokens = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    return tokens

def apply_tokenization(df, tokenizer, text_column="cleaned_content"):
    """
    Apply the tokenize_text function to the specified text column of a DataFrame.
    """
    df['tokens'] = df[text_column].apply(lambda x: tokenize_text(x, tokenizer))
    return df

def split_data(df, test_size=0.2, random_state=42):
    """
    Splits the DataFrame into training and validation sets.
    """
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_df, val_df
