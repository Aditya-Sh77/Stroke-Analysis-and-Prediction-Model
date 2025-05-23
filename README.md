
# 🧠 Stroke Prediction using Machine Learning

This project uses a healthcare dataset to predict the likelihood of a stroke based on medical and demographic features.

## 📊 Features
- Cleaned and visualized real-world stroke data
- Balanced class distribution using SMOTE
- Trained Logistic Regression, Random Forest, and XGBoost classifiers
- Achieved ~96% ROC-AUC with Random Forest
- Used SHAP for feature interpretability

## 🔧 Tech Stack
- Python, pandas, numpy
- Scikit-learn, imbalanced-learn
- XGBoost, SHAP
- Matplotlib, Seaborn

## 📈 Results
- Random Forest outperformed other models
- Age, average glucose level, and BMI were top stroke predictors

## 📁 Files
- `stroke_prediction.ipynb`: Full analysis and ML model
- `requirements.txt` : Required Libraries
- `healthcare-dataset-stroke-data.csv`: Dataset (publicly available from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset))

## 🧠 Future Work
- Deploy model using Streamlit
- Add cross-validation and feature selection
- Expand to time-series or live input predictions

## ✅ How to Run
```bash
pip install -r requirements.txt
jupyter notebook
