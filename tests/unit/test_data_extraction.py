import pandas as pd
import pytest
from src.data_extraction import load_data

def test_load_data_success():
    """Test that the load_data function loads the data correctly."""
    file_path = 'dataset.csv'
    try:
        df = load_data(file_path)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
    except FileNotFoundError:
        pytest.fail(f"The file {file_path} was not found.")


def test_load_data_file_not_found():
    """Test that the load_data function raises FileNotFoundError for a missing file."""
    file_path = 'non_existent_file.csv'
    with pytest.raises(FileNotFoundError):
        load_data(file_path)


def test_load_data_empty_file():
    """Test that the load_data function handles empty files correctly."""
    # Create an empty CSV file for testing
    with open('empty_file.csv', 'w') as f:
        pass  # Create an empty file
    
    df = load_data('empty_file.csv')
    assert df is None

def test_load_data_expected_columns():
    """Test that the loaded DataFrame has the expected columns."""
    file_path = 'dataset.csv'
    df = load_data(file_path)
    expected_columns = ['reviewId', 'userName', 'userImage', 'content', 'score', 'thumbsUpCount', 'reviewCreatedVersion', 'at', 'replyContent', 'repliedAt', 'sortOrder', 'appId']  # Remplace avec les noms de tes colonnes
    assert list(df.columns) == expected_columns
