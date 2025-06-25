# src/ingest.py

import pandas as pd
import os

def load_data(path: str = "data/bank.csv") -> pd.DataFrame:
    """
    Load the Bank Marketing dataset.

    Parameters:
    - path (str): File path to the CSV file

    Returns:
    - DataFrame: Loaded dataset
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    
    df = pd.read_csv(path, sep=';')  # Bank Marketing uses ; as delimiter
    return df
