# src/sampler.py

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def apply_smote(X: pd.DataFrame, y: pd.Series, sampling_strategy: float = 0.5, random_state: int = 42):
    """
    Apply SMOTE oversampling to balance the minority class.

    Parameters:
    - X (pd.DataFrame): Feature dataframe
    - y (pd.Series): Target variable
    - sampling_strategy (float): Desired ratio of minority to majority class after resampling
    - random_state (int): Random seed for reproducibility

    Returns:
    - X_resampled (pd.DataFrame)
    - y_resampled (pd.Series)
    """
    sm = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_res, y_res = sm.fit_resample(X, y)
    
    return X_res, y_res

def train_val_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """
    Split dataset into training and validation sets.

    Returns:
    - X_train, X_val, y_train, y_val
    """
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
