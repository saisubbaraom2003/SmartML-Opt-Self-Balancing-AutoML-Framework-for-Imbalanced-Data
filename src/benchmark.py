# src/benchmark.py
from lightautoml.tasks import Task
from sklearn.metrics import classification_report, f1_score, confusion_matrix, roc_curve, auc # Added confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import time
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
from tpot import TPOTClassifier
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns
import os 



def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    plt.close()

def plot_roc_curve(y_true, y_proba, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    plt.close()

def run_tpot(X, y):
    print("Running TPOT...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y) # stratify=y is good!
    model = TPOTClassifier(generations=5, population_size=20, verbosity=2, n_jobs=-1) # No explicit scoring
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print("TPOT F1 Score:", f1_score(y_test, y_pred))
    print("Time:", round(end - start, 2), "sec")
    print(classification_report(y_test, y_pred))


    os.makedirs("results", exist_ok=True)
    

    plot_confusion_matrix(y_test, y_pred, save_path=r"C:\Users\saisu\OneDrive\Desktop\GitHub Repos\SmartML-Opt-Self-Balancing-AutoML-Framework-for-Imbalanced-Data\Visualisations/tpot_confusion_matrix.png")
    print("TPOT Confusion Matrix saved to results/tpot_confusion_matrix.png")

    plot_roc_curve(y_test, y_proba, save_path=r"C:\Users\saisu\OneDrive\Desktop\GitHub Repos\SmartML-Opt-Self-Balancing-AutoML-Framework-for-Imbalanced-Data\Visualisations/tpot_roc_curve.png")
    print("TPOT ROC Curve saved to results/tpot_roc_curve.png")


def run_lightautoml(X, y):
    task = Task('binary') 

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    if isinstance(y, np.ndarray):
        y = pd.Series(y, name="target") 
    train_data = X.copy()
    train_data['target'] = y
    
    automl = TabularAutoML(task=task, timeout=300, cpu_limit=4)
    start = time.time()
    

    oof_pred = automl.fit_predict(train_data, roles={"target": "target"})
    

    preds_proba_oof = oof_pred.data[:, 0] if oof_pred.data.ndim > 1 else oof_pred.data
    preds_oof = (preds_proba_oof > 0.5).astype(int)

   
    
    print("LightAutoML Classification Report (Out-of-Fold Predictions):")
    print(classification_report(y, preds_oof))

    end = time.time()
    print(f"LightAutoML training time: {end - start:.2f} seconds")

    os.makedirs("results", exist_ok=True)

    plot_confusion_matrix(y, preds_oof, save_path=r"C:\Users\saisu\OneDrive\Desktop\GitHub Repos\SmartML-Opt-Self-Balancing-AutoML-Framework-for-Imbalanced-Data\Visualisations/lightautoml_confusion_matrix.png")
    print("LightAutoML Confusion Matrix saved to results/lightautoml_confusion_matrix.png")

    plot_roc_curve(y, preds_proba_oof, save_path=r"C:\Users\saisu\OneDrive\Desktop\GitHub Repos\SmartML-Opt-Self-Balancing-AutoML-Framework-for-Imbalanced-Data\Visualisations/lightautoml_roc_curve.png")
    print("LightAutoML ROC Curve saved to results/lightautoml_roc_curve.png")
