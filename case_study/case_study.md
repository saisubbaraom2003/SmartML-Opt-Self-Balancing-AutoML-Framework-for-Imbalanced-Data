# Case Study: SmartML-Opt – A Self-Balancing AutoML Framework for Imbalanced Classification

## 1. Problem Statement and Objectives

In many real-world applications like marketing campaigns, fraud detection, and medical diagnosis, datasets suffer from **class imbalance** — where one class dominates the other. This leads to biased models that perform poorly on the minority class.

**Objective:**  
To develop a robust AutoML pipeline that integrates **dynamic data balancing, feature selection**, and **automated model tuning** to address imbalanced classification tasks.

---

## 2. Dataset: Bank Marketing Data

- Source: UCI Bank Marketing Dataset
- Type: Tabular
- Target: `y` (whether the client subscribed to a term deposit)
- Class Distribution:
  - Yes: ~11.7%
  - No: ~88.3%

---

## 3. Data Preprocessing Steps

- Remove missing values and "unknown" labels.
- Encode categorical variables using LabelEncoder.
- Scale numeric variables using StandardScaler.
- Apply **SMOTE** to balance the target class.

---

## 4. Model Development

### Steps:
- Use **Recursive Feature Elimination (RFE)** for feature selection.
- Apply **SMOTE** to generate synthetic minority samples.
- Run **GridSearchCV** across 3 models:
  - Random Forest
  - Gradient Boosting
  - Support Vector Machine

### Final Best Model:
- Selected: Random Forest with tuned hyperparameters.
- Optimized for **F1-score** and **recall**.

---

## 5. Visualizations & Insights

| Metric        | Value |
|---------------|-------|
| Validation F1 | 0.71  |
| Recall        | 0.68  |
| AUC           | 0.82  |

### ✅ Key Visuals:
- Confusion Matrix
- ROC Curve
- Feature Importance Plot (optional)

---

## 6. Recommendations

- Use this framework for any **imbalanced classification problem**.
- Integrate it into existing ML pipelines to **auto-tune + balance + select features**.
- Can be extended to add **LightAutoML / Optuna / Green-AutoML** for optimization.

---

## 7. Conclusion

SmartML-Opt effectively automates the critical steps in model development for imbalanced datasets. By integrating data sampling, feature selection, model tuning, and experiment tracking, it enables accurate and reproducible ML workflows with minimal manual intervention.

