# src/ingest.py

import pandas as pd
import os

def load_data(path: str = "data/bank.csv") -> pd.DataFrame:

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    
    df = pd.read_csv(path, sep=';')  
    return df
