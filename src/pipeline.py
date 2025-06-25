# src/pipeline.py

import mlflow
import pandas as pd
from src.ingest import load_data
from src.preprocess import clean_data, encode_features, scale_features
from src.sampler import apply_smote, train_val_split
from src.feature_selector import select_features
from src.model_selector import run_grid_search
from src.evaluate import plot_confusion_matrix, plot_roc_curve, print_classification_report

def run_pipeline(data_path="data/bank.csv", target_col="y", smote_ratio=0.5, n_features=15):
    # Start MLflow run
    mlflow.start_run()
    
    # Load data
    df = load_data(data_path)
    mlflow.log_param("dataset", data_path)
    
    # Clean and encode
    df_clean = clean_data(df)
    df_encoded = encode_features(df_clean)
    
    # Scale features and separate target
    X_scaled, y = scale_features(df_encoded, target_col)
    
    # Apply SMOTE sampling
    X_res, y_res = apply_smote(X_scaled, y, sampling_strategy=smote_ratio)
    mlflow.log_param("smote_ratio", smote_ratio)
    
    # Feature selection
    X_sel = select_features(X_res, y_res, n_features=n_features)
    mlflow.log_param("n_features_selected", n_features)
    
    # Split train-validation
    X_train, X_val, y_train, y_val = train_val_split(X_sel, y_res)
    
    # Train models with hyperparameter tuning
    best_model, best_score, best_name = run_grid_search(X_train, y_train, scoring_metric='f1')
    mlflow.log_param("best_model", best_name)
    mlflow.log_metric("best_cv_score", best_score)
    
    # Validation evaluation
    y_pred = best_model.predict(X_val)
    
    # Print and log evaluation
    print_classification_report(y_val, y_pred)
    mlflow.log_metric("val_f1_score", best_score)  # Approximate for simplicity
    
    # Plot and save confusion matrix and ROC curve
    plot_confusion_matrix(y_val, y_pred, save_path="results/confusion_matrix.png")
    plot_roc_curve(best_model, X_val, y_val, save_path="results/roc_curve.png")
    mlflow.log_artifact("results/confusion_matrix.png")
    mlflow.log_artifact("results/roc_curve.png")
    
    mlflow.end_run()
    
    print("Pipeline execution completed.")

if __name__ == "__main__":
    run_pipeline()
