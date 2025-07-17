import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()
    
 
    df.replace('unknown', pd.NA, inplace=True)
    df.dropna(inplace=True)
    
    return df.reset_index(drop=True)

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()
   
    cat_cols = df.select_dtypes(include='object').columns
    le = LabelEncoder()
    
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
    
    return df

def scale_features(df: pd.DataFrame, target_column: str) -> tuple:
 
    X = df.drop(columns=[target_column])
    y = df[target_column]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X_scaled, y

def map_target_variable(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    
    df = df.copy()
    df[target_column] = df[target_column].map({'yes': 1, 'no': 0})
    return df
