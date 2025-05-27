# Diabetes Prediction Using Machine Learning

This project is a comprehensive machine learning pipeline developed for predicting diabetes using an imbalanced medical dataset of 100,000 patient records. It was built as part of the CSE422 (Artificial Intelligence) lab project at BRAC University.

## 📊 Dataset
- Source: [Kaggle - Diabetes Prediction Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)
- Total records: 100,000
- Features: 9 clinical and demographic features (including age, BMI, glucose level, etc.)
- Target: Diabetes (binary classification: 0 - non-diabetic, 1 - diabetic)
- Imbalance: ~11:1 ratio (non-diabetic:diabetic)

## 🧪 Models Used
Five supervised ML models were evaluated:
- Logistic Regression
- Decision Tree Classifier
- K-Nearest Neighbors (KNN)
- Random Forest Classifier
- Multi-layer Perceptron (Neural Network)

## ⚙️ Preprocessing
- Missing value imputation
- Feature scaling using `StandardScaler`
- Categorical encoding with `OneHotEncoder`
- Feature engineering: BMI categories, age groups
- No oversampling used to preserve class distribution

## 🔍 Performance (Weighted Averages)
| Model              | Accuracy | F1-Score | AUC    |
|-------------------|----------|----------|--------|
| Decision Tree      | 0.9670   | 0.96     | 0.9636 |
| Random Forest      | 0.9664   | 0.96     | 0.9622 |
| Logistic Regression| 0.9554   | 0.95     | 0.9438 |
| Neural Network     | 0.9563   | 0.95     | 0.9435 |
| KNN                | 0.9553   | 0.95     | 0.8974 |

## 📌 Key Insights
- Tree-based models (Decision Tree, Random Forest) outperformed others, especially in AUC and F1-score.
- KNN underperformed in separating classes, showing lower AUC despite high accuracy.
- Clinical indicators like HbA1c > 6.5%, glucose > 140 mg/dL, and BMI > 30 strongly correlated with diabetes.


## 🛠️ Tools & Libraries
- Python
- Scikit-learn
- Pandas, NumPy, Matplotlib, Seaborn
- Jupyter Notebook

## 📚 Authors
**Md. Imam Hasan**  
Dept. of Computer Science & Engineering  
BRAC University  
📧 md.imam.hasan@g.bracu.ac.bd
