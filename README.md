# 🩺 Diabetes Prediction & Diagnosis: A Comparative ML Study

This repository contains a comprehensive machine learning analysis for predicting diabetes diagnosis based on patient demographics and clinical history. Six distinct ML classifiers were trained, optimized, and evaluated to determine the most effective diagnostic model.

---

## 📁 Project Structure

The project code is organized inside the [Diabetes_Prediction](file:///c:/Users/Asus/Desktop/DiabetesProject/Diabetes_Prediction) repository root:

*   **[CODES/](file:///c:/Users/Asus/Desktop/DiabetesProject/Diabetes_Prediction/CODES)**: Contains Jupyter notebooks implementing the data pipeline, training, and model tuning:
    *   **[IT24102571_XGBoost (1).ipynb](file:///c:/Users/Asus/Desktop/DiabetesProject/Diabetes_Prediction/CODES/IT24102571_XGBoost%20(1).ipynb)**: Extreme Gradient Boosting model training and hyperparameter optimization.
    *   **[IT24102584_LogisticRegressionFinal.ipynb](file:///c:/Users/Asus/Desktop/DiabetesProject/Diabetes_Prediction/CODES/IT24102584_LogisticRegressionFinal.ipynb)**: Baseline and tuned Logistic Regression models with ROC-AUC analysis.
    *   **[IT24102633_MLP.ipynb](file:///c:/Users/Asus/Desktop/DiabetesProject/Diabetes_Prediction/CODES/IT24102633_MLP.ipynb)**: Multi-Layer Perceptron (MLP) artificial neural network classifier.
    *   **[IT24102679_DECISION_TREE_.ipynb](file:///c:/Users/Asus/Desktop/DiabetesProject/Diabetes_Prediction/CODES/IT24102679_DECISION_TREE_.ipynb)**: Decision Tree classifier with tree depth pruning to prevent overfitting.
    *   **[IT24102687_KNN.ipynb](file:///c:/Users/Asus/Desktop/DiabetesProject/Diabetes_Prediction/CODES/IT24102687_KNN.ipynb)**: K-Nearest Neighbors (KNN) baseline and optimized model with hyperparameter search.
    *   **[train_random_forest.ipynb](file:///c:/Users/Asus/Desktop/DiabetesProject/Diabetes_Prediction/CODES/train_random_forest.ipynb)**: Random Forest model training, validation, and PCA feature exploration.
*   **[.gitattributes](file:///c:/Users/Asus/Desktop/DiabetesProject/Diabetes_Prediction/.gitattributes)**: Git settings for repository file attributes.
*   **[README.md](file:///c:/Users/Asus/Desktop/DiabetesProject/Diabetes_Prediction/README.md)**: Main project documentation and model evaluation summary.

---

## 📊 Dataset & Preprocessing

The model uses the preprocessed dataset `diabetes_data_pre_processed.csv` consisting of the following key features:
*   **Demographic Data**: Gender, Age, and Age Group categories (`Young`, `Adult`, `Middle-aged`, `Senior`).
*   **Clinical History**: Hypertension, Heart Disease, and Smoking History.
*   **Diagnostic Metrics**: Body Mass Index (BMI), HbA1c Level, and Blood Glucose Level.
*   **Target Column**: `diabetes` (1 = Diabetic, 0 = Non-Diabetic).

### Preprocessing Steps Applied:
1.  **Stratified Split**: Train-Test split (80/20 ratio) stratified to maintain identical diabetes class proportions in both sets.
2.  **One-Hot Encoding**: Handled categorical features (e.g., age groups, smoking history).
3.  **Feature Scaling**: Applied `StandardScaler` to ensure numerical features possess a mean of 0 and variance of 1.

---

## 🏆 Model Evaluation & Comparison

For clinical diagnostics, minimizing False Negatives is critical to avoid leaving patients undiagnosed. Therefore, **Recall** and **F1 Score** are prioritized over simple Accuracy.

Below is a summary of the baseline vs. tuned model performance metrics across the test split:

| Model | Baseline F1 Score | Tuned F1 Score | Best Accuracy / AUC | Hyperparameter Tuning Details |
| :--- | :---: | :---: | :---: | :--- |
| **Random Forest** | `0.9930` | `0.9928` | — | PCA feature reduction (`PC1`, `PC2`), F1 macro optimization. |
| **K-Nearest Neighbors (KNN)** | `0.9608` | `0.9831` | — | Optimized search for optimal neighbor count `k`. |
| **XGBoost Classifier** | `0.9408` | `0.9221` | — | Grid search tuning optimized using the F1 metric. |
| **Decision Tree** | `0.9900` | `0.9000` | `89.63%` Acc (Tuned) / `0.9692` AUC | Pruned decision boundaries via constrained depth (`max_depth=10`). |
| **Multi-Layer Perceptron (MLP)** | `0.9102` | `0.9043` | — | Feed-forward neural network trained with backpropagation. |
| **Logistic Regression** | `0.8708` | `0.8725` | `87.08%` Acc / `0.9488` AUC | Linear classifier baseline optimized for `C` regularization. |

> [!IMPORTANT]
> The baseline Decision Tree and Random Forest models showed near-perfect metrics (~99% F1) on the test split due to model complexity. However, tuning constraints (such as setting `max_depth=10` on the Decision Tree) were applied to ensure model generalization and avoid overfitting to the training split.

> [!TIP]
> Use the **GridSearchCV** sections in each notebook to re-run hyperparameter optimization with different parameters or search ranges.

---

## 🚀 Getting Started

### Prerequisites
Make sure you have Python 3.8+ installed along with the required libraries:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost
```

### Running the Analysis
1. Navigate to the repository root directory:
   ```bash
   cd Diabetes_Prediction
   ```
2. Launch Jupyter Notebook or JupyterLab:
   ```bash
   jupyter notebook
   ```
3. Open any of the files in the [CODES/](file:///c:/Users/Asus/Desktop/DiabetesProject/Diabetes_Prediction/CODES) directory to explore the step-by-step implementation, visualization, and validation metrics.
