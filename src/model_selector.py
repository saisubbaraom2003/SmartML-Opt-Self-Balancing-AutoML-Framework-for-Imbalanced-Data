# src/model_selector.py

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, f1_score, recall_score, precision_score

def get_model_grid():
 
    models = []
    
    # Random Forest
    rf = RandomForestClassifier(random_state=42)
    rf_params = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5]
    }
    models.append(('RandomForest', rf, rf_params))
    
    # Gradient Boosting
    gb = GradientBoostingClassifier(random_state=42)
    gb_params = {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    }
    models.append(('GradientBoosting', gb, gb_params))
    
    # Support Vector Classifier
    svc = SVC(random_state=42)
    svc_params = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    models.append(('SVC', svc, svc_params))
    
    return models

def run_grid_search(X, y, scoring_metric='f1'):

    models = get_model_grid()
    best_score = 0
    best_model = None
    best_name = None
    
    scorer = make_scorer(f1_score) if scoring_metric == 'f1' else make_scorer(recall_score)
    
    for name, model, params in models:
        print(f"Training {name}...")
        grid = GridSearchCV(model, params, scoring=scorer, cv=5, n_jobs=-1)
        grid.fit(X, y)
        
        if grid.best_score_ > best_score:
            best_score = grid.best_score_
            best_model = grid.best_estimator_
            best_name = name
    
    print(f"Best model: {best_name} with score: {best_score:.4f}")
    return best_model, best_score, best_name
