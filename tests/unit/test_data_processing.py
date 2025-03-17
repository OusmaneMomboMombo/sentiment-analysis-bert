import pandas as pd
import pytest
from src.data_processing import clean_text, apply_cleaning

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