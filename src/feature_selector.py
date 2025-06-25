# src/feature_selector.py

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def select_features(X: pd.DataFrame, y: pd.Series, n_features: int = 15, estimator=None):

    if estimator is None:
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
    
    selector = RFE(estimator, n_features_to_select=n_features, step=1)
    selector = selector.fit(X, y)
    
    selected_cols = X.columns[selector.support_]
    return X[selected_cols]
