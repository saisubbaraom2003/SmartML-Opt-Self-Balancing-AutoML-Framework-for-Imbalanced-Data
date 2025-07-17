# src/pipeline.py


import pandas as pd
from src.ingest import load_data
from src.preprocess import clean_data, encode_features, scale_features
from src.sampler import apply_smote, train_val_split
from src.feature_selector import select_features
from src.model_selector import run_grid_search
from src.evaluate import plot_confusion_matrix, plot_roc_curve, print_classification_report

def run_pipeline(data_path="data/bank.csv", target_col="y", smote_ratio=0.5, n_features=15):
   
    df = load_data(data_path)
    

    df_clean = clean_data(df)
    df_encoded = encode_features(df_clean)
   
    X_scaled, y = scale_features(df_encoded, target_col)
    
  
    X_res, y_res = apply_smote(X_scaled, y, sampling_strategy=smote_ratio)


    X_sel = select_features(X_res, y_res, n_features=n_features)
  
    
    X_train, X_val, y_train, y_val = train_val_split(X_sel, y_res)

    best_model, best_score, best_name = run_grid_search(X_train, y_train, scoring_metric='f1')
   
    

    y_pred = best_model.predict(X_val)

    print_classification_report(y_val, y_pred)
    

    plot_confusion_matrix(y_val, y_pred, save_path=r"C:\Users\saisu\OneDrive\Desktop\GitHub Repos\SmartML-Opt-Self-Balancing-AutoML-Framework-for-Imbalanced-Data\Visualisations/confusion_matrix.png")
    plot_roc_curve(best_model, X_val, y_val, save_path=r"C:\Users\saisu\OneDrive\Desktop\GitHub Repos\SmartML-Opt-Self-Balancing-AutoML-Framework-for-Imbalanced-Data\Visualisations/roc_curve.png")
    
    
    print("Pipeline execution completed.")

if __name__ == "__main__":
    run_pipeline()
