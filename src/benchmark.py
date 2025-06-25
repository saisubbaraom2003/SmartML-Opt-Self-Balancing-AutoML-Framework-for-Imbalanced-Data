# src/benchmark.py

from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier
from autosklearn.classification import AutoSklearnClassifier
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")

def run_tpot(X, y):
    print("Running TPOT...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
    model = TPOTClassifier(generations=5, population_size=20, verbosity=2, n_jobs=-1)
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    y_pred = model.predict(X_test)
    print("TPOT F1 Score:", f1_score(y_test, y_pred))
    print("Time:", round(end - start, 2), "sec")
    print(classification_report(y_test, y_pred))

def run_autosklearn(X, y):
    print("Running AutoSklearn...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
    model = AutoSklearnClassifier(time_left_for_this_task=300, per_run_time_limit=60, n_jobs=-1)
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    y_pred = model.predict(X_test)
    print("AutoSklearn F1 Score:", f1_score(y_test, y_pred))
    print("Time:", round(end - start, 2), "sec")
    print(classification_report(y_test, y_pred))

def run_lightautoml(X, y):
    from lightautoml.automl.presets.tabular_presets import TabularAutoML
    from lightautoml.tasks import Task
    import numpy as np
    from sklearn.model_selection import train_test_split
    print("Running LightAutoML...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
    task = Task('binary')
    automl = TabularAutoML(task=task, timeout=300)
    import datatable as dt
    train = dt.Frame(X_train)
    train['target'] = dt.Frame(y_train.values)
    test = dt.Frame(X_test)
    start = time.time()
    oof_pred = automl.fit_predict(train, roles={'target': 'target'})
    preds = automl.predict(test).to_pandas().values[:, 0]
    preds = (preds > 0.5).astype(int)
    end = time.time()
    print("LightAutoML F1 Score:", f1_score(y_test, preds))
    print("Time:", round(end - start, 2), "sec")
    print(classification_report(y_test, preds))
