# Sentiment Analysis with BERT

This project implements a sentiment analysis pipeline using a pre-trained BERT model. It allows you to analyze the sentiment (positive or negative) of text reviews.

## Overview

The pipeline consists of three main components:

1.  **Data Extraction:** Loading and preparing raw text data from a CSV file.
2.  **Data Processing:** Cleaning, tokenizing, and preparing the text data for the BERT model. This includes removing unnecessary characters, lower-casing, and converting text into the format expected by the BERT tokenizer.
3.  **Model Training & Inference:** Fine-tuning a pre-trained BERT model for sentiment classification and creating an inference script to predict the sentiment of new text.

## Repository Structure

```
.
├── src/
│   ├── data_extraction.py  # Loads data from CSV file
│   ├── data_processing.py  # Cleans, tokenizes, and splits data
│   ├── model.py            # Fine-tunes and saves the BERT model
│   └── inference.py        # Predicts sentiment of new text
├── tests/
│   └── unit/
│       ├── test_data_extraction.py  # Tests data extraction
│       ├── test_data_processing.py  # Tests data processing
│       ├── test_model.py            # Tests model training
│       └── test_inference.py        # Tests inference
├── requirements.txt      # Lists project dependencies
└── README.md             # This file
```

## Setup Instructions

1.  **Clone the repository:**

    ```
    git clone 
    cd 
    ```

2.  **Create a virtual environment (recommended):**

    ```
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install dependencies:**

    ```
    pip install -r requirements.txt
    ```

## Usage

1.  **Prepare your data:**

    *   Ensure your data is in a CSV file named `dataset.csv` (or modify `data_extraction.py` to load from a different file).
    *   The CSV should have columns named `"content"` (for the text review) and `"score"` (for the sentiment score).

2.  **Train the model (optional):**

    *   Run the `model.py` script to fine-tune the BERT model:

        ```
        python src/model.py
        ```

        This will train the model and save the trained weights to `saved_models/bert_sentiment.pth`. You only need to do this if you want to train the model yourself. Otherwise, you can use a pre-trained model.

3.  **Run inference:**

    *   Run the `inference.py` script to predict the sentiment of sample texts:

        ```
        python src/inference.py
        ```

        This will load the trained model, tokenize the texts, and print the predicted sentiment for each.  Modify the `texts` list in `inference.py` to test with your own examples.

## Project Structure Details

*   `src/data_extraction.py`: Contains the `load_data` function, which is responsible for reading the dataset from a CSV file into a pandas DataFrame.  Handles `FileNotFoundError` and returns `None` for empty files.
*   `src/data_processing.py`: Contains functions for cleaning, tokenizing, and splitting data into training and validation sets:
    *   `clean_text`: Removes unnecessary characters, converts to lowercase, etc.
    *   `apply_cleaning`: Applies the cleaning function to a DataFrame.
    *   `apply_tokenization`: Tokenizes text using the Hugging Face `AutoTokenizer`.
    *   `split_data`: Splits the data into training and validation sets.
*   `src/model.py`: Contains functions for loading, training, evaluating, and saving the BERT model:
    *   `load_model`: Loads a pre-trained BERT model for sequence classification.
    *   `train_model`: Fine-tunes the model on the sentiment dataset.
    *   `evaluate_model`: Evaluates the trained model on the validation dataset.
    *   `save_model`: Saves the trained model to a file.
    *   `load_trained_model`: Loads a trained model from a file.
    *   `SentimentDataset`: A PyTorch `Dataset` class for handling sentiment data.
*   `src/inference.py`: Contains the `predict_sentiment` function for predicting the sentiment of new text:
    *   Loads the trained model and tokenizer.
    *   Takes a text input.
    *   Returns "positive" or "negative" sentiment.

## Collaboration

This project was a collaborative effort:

*   Student A: Ousmane MOMBO MOMBO - @OusmaneMomboMombo
*   Student B: REGOMBY Nathan - @AdrielNathan

Collaboration was achieved through:

*   Branching
*   Pull requests
*   Code reviews
*   Shared continuous integration (CI) setup
