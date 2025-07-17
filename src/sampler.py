# src/sampler.py

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def apply_smote(X: pd.DataFrame, y: pd.Series, sampling_strategy: float = 0.5, random_state: int = 42):
    
    sm = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_res, y_res = sm.fit_resample(X, y)
    
    return X_res, y_res

def train_val_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
