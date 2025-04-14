"""
Module for loading CSV data into a pandas DataFrame.
"""

import os
import pandas as pd

def load_data(file_path):
    """
    Loads data from a CSV file into a pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame or None: The loaded DataFrame, or None if an error occurred.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at {file_path}")
    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully!")
        return df
    except pd.errors.EmptyDataError:
        print(f"Error: The file {file_path} is empty.")
        return None
    except pd.errors.ParserError as e:
        print(f"Parsing error while reading {file_path}: {e}")
        return None
    except OSError as e:
        print(f"OS error while reading {file_path}: {e}")
        return None
