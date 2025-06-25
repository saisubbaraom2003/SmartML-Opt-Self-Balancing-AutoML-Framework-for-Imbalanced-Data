# src/preprocess.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the dataset by removing unknown values and standardizing formats.

    Parameters:
    - df (pd.DataFrame): Raw dataframe

    Returns:
    - pd.DataFrame: Cleaned dataframe
    """
    # Drop rows with 'unknown' values
    df.replace('unknown', pd.NA, inplace=True)
    df.dropna(inplace=True)
    
    return df.reset_index(drop=True)

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label encodes categorical features.

    Parameters:
    - df (pd.DataFrame): Cleaned dataframe

    Returns:
    - pd.DataFrame: Encoded dataframe
    """
    cat_cols = df.select_dtypes(include='object').columns
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
    
    return df

def scale_features(df: pd.DataFrame, target_column: str) -> tuple:
    """
    Scales numeric features using StandardScaler.

    Parameters:
    - df (pd.DataFrame): Encoded dataframe
    - target_column (str): Name of target column

    Returns:
    - X_scaled (pd.DataFrame): Scaled features
    - y (pd.Series): Target variable
    """
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X_scaled, y
