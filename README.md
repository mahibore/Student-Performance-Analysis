# Student-Performance-Analysis
A machine learning project that analyzes student data to identify factors affecting academic performance. Includes data cleaning, visualization, and model building using algorithms like Random Forest to predict scores. Helps in making data-driven educational decisions.
# 🎓 Student Performance Analysis using Machine Learning

This project explores the impact of academic and lifestyle factors on student performance using a real-world dataset. We apply machine learning to predict student success and visualize key insights.

## 📌 Objective

- Analyze student data including scores, study habits, and preparation
- Discover patterns and correlations between variables
- Predict student performance using regression models

## 🗃️ Dataset

The dataset includes the following columns:
- Hours Studied
- Previous Scores
- Extracurricular Activities
- Sleep Hours
- Sample Question Papers Practiced
- Performance Index (Target)

## 📊 Technologies Used

- Python (Pandas, NumPy)
- Seaborn & Matplotlib for visualization
- Scikit-learn for ML model building
- Jupyter Notebook

## 📈 Workflow

1. **Data Import & Cleaning**
2. **Exploratory Data Analysis (EDA)** with heatmaps and pairplots
3. **Feature Selection**
4. **Model Building** using Linear Regression
5. **Evaluation** using Mean Squared Error & R² Score

## 🔍 Sample Code

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
