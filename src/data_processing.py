import pandas as pd
import re

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

# Example usage (for testing purposes)
if __name__ == '__main__':
    data = {'content': ["This is a TeXt with  some   SPACES and 123! #@$"]}
    example_df = pd.DataFrame(data)
    
    cleaned_df = apply_cleaning(example_df)
    print(cleaned_df)
