# SmartML-Opt – A Self-Balancing AutoML Framework for Imbalanced Classification

##  Project Description

SmartML-Opt is an end-to-end AutoML pipeline designed to tackle **imbalanced classification problems** using a self-balancing strategy. The framework integrates:

- Dynamic oversampling (SMOTE)
- Feature selection (RFE)
- Model tuning (GridSearchCV)

It is tested on the **UCI Bank Marketing Dataset** and optimized for **F1-score, Precision, recall, and AUC**.

---

##  Dataset Used

- **Source:** [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
- **Type:** Tabular
- **Target Variable:** `y` (binary: yes/no)
- **Imbalance Ratio:** Yes ~11.7%, No ~88.3%

---

##  Pipeline Stages

1. **Data Cleaning**  
   - Remove missing or 'unknown' entries.

2. **Label Encoding**  
   - Convert categorical features using `LabelEncoder`.

3. **Feature Scaling**  
   - Apply `StandardScaler` to numerical columns.

4. **SMOTE Oversampling**  
   - Dynamically balance dataset with SMOTE.

5. **Feature Selection**  
   - Use `Recursive Feature Elimination (RFE)` with base model.

6. **Model Training and Hyperparameter Tuning**  
   - Models: Random Forest, Gradient Boosting, SVM  
   - Optimized using `GridSearchCV` for F1-score.

7. **Evaluation**  
   - Metrics: F1, Recall,Precision, AUC  
   - Confusion Matrix and ROC Curve.
.

---

##  Final Best Model

- **Model Selected:** Gradient Boosting 
- **Validation Performance:**
  - F1 Score `0.79`
  - Recall `0.85`
  - Precision `0.75`
  - AUC Score `0.94`

---

##  Visualizations Included

- Confusion Matrix  
- ROC Curve  
- Categorical Features vs Target
- Distribution of Continuous Features by Target  
- Architecture Pipeline Flow

---

##  Requirements

- `pandas`
-`numpy`
-`scikit-learn`
-`imbalanced-learn`
-`matplotlib`
-`seaborn`
-`tpot`
-`auto-sklearn`
-`lightautoml`
-`jupyter`


---

##  How to Run

### 1. Clone Repository

### 2. Install Dependencies

### 3. Run Jupyter Notebook

##  Output Artifacts

- Evaluation plots (confusion matrix, ROC)

##  Recommendations

- Use SmartML-Opt as a template for any **imbalanced classification** task.
- Extend the pipeline with:
  - **LightAutoML**
  - **Optuna for faster hyperparameter tuning**
  - **Green-AutoML** for efficiency

---

##  License

"Aligned with Megaminds IT Services' R&D goals, this project delivers a robust AutoML framework addressing real-world imbalanced classification challenges using dynamic sampling and modular ML design."

---

##  Author

**Name:** Sai Subba Rao Mahendrakar  

**GitHub:** [saisubbaraom2003]([https://github.com/saisubba13](https://github.com/saisubbaraom2003))  

**LinkedIn:** [Sai Subba Rao Mahendrakar]([https://linkedin.com/in/saisubbarao](https://www.linkedin.com/in/sai-subba-rao-mahendrakar-934b9a334))  

**Portfolio** [Sai Subba Rao Mahendrakar](https://mahendrakar.netlify.app)      

**Gmail** [sai.subbu.in@gmail.com](mahendrakar2003@gmail.com)
