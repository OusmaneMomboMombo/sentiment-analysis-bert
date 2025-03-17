import unittest
import pandas as pd
import os
from src.data_extraction import load_data

class TestDataExtraction(unittest.TestCase):

    def test_load_data_success(self):
        df = load_data('dataset.csv')
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)  
        self.assertIn('reviewId', df.columns)  
        self.assertIn('userName', df.columns)  
        self.assertIn('content', df.columns)   
        self.assertIn('score', df.columns)     

    def test_load_data_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_data('non_existent_file.csv')

    def test_load_data_empty_file(self):
        with open('empty_file.csv', 'w'):
            pass  # Crée un fichier vide

        df = load_data('empty_file.csv')
        self.assertIsNone(df)

        os.remove('empty_file.csv')

    def tearDown(self):
        if os.path.exists('empty_file.csv'):
            os.remove('empty_file.csv')
        if os.path.exists('test_data.csv'):
            os.remove('test_data.csv')

if __name__ == '__main__':
    unittest.main()
