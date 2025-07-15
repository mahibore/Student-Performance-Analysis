# Student Performance Analysis

This repository contains a Jupyter Notebook for analyzing student performance. The analysis aims to predict student GPA based on various factors.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Code Example](#code-example)
- [Predicted Output](#predicted-output)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project explores student academic performance using data analysis and machine learning. The primary goal is to build a model that can predict a student's Grade Point Average (GPA) based on input features such as study hours, previous GPA, and extracurricular activities.

## Features

- Data loading and initial exploration.
- Data preprocessing (handling missing values, encoding categorical features, scaling).
- Model training (Linear Regression is used in the provided example).
- Prediction of GPA based on new input data.

## Installation

To run this notebook locally, you'll need Python and Jupyter.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/student-performance-analysis.git
    cd student-performance-analysis
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn warnings
    ```
    *(Note: The `warnings` library is typically built-in and doesn't need explicit installation.)*

## Usage

1.  **Start Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```

2.  **Open the notebook:**
    Navigate to `Student Performance Analysis.html` (or the original `.ipynb` if you have it) in your Jupyter interface and open it.

3.  **Run the cells:**
    Execute the cells sequentially to see the data loading, preprocessing, model training, and prediction steps.

## Code Example

Here's a snippet demonstrating the prediction part of the code:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Dummy data for demonstration (replace with your actual dataset)
data = {
    'Study_Hours': [5, 10, 3, 8, 6, 12, 7, 4, 9, 11],
    'Previous_GPA': [3.0, 3.5, 2.5, 3.8, 3.2, 3.9, 3.1, 2.8, 3.7, 3.6],
    'Extracurricular_Activities': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No'],
    'Attendance': [90, 95, 80, 92, 88, 98, 85, 75, 93, 96],
    'GPA': [3.2, 3.8, 2.7, 4.0, 3.4, 4.0, 3.3, 2.9, 3.9, 3.7]
}
df = pd.DataFrame(data)

# Define features (X) and target (y)
X = df[['Study_Hours', 'Previous_GPA', 'Extracurricular_Activities', 'Attendance']]
y = df['GPA']

# Define categorical and numerical features
categorical_features = ['Extracurricular_Activities']
numerical_features = ['Study_Hours', 'Previous_GPA', 'Attendance']

# Create a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Create a pipeline with preprocessing and a Linear Regression model
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', LinearRegression())])

# Split data (for training, though this example uses full data for simplicity)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Example prediction
new_student_data = pd.DataFrame([[10, 3.9, 'Yes', 95]], columns=X.columns)
predicted_gpa = model.predict(new_student_data)

print(f"The predicted GPA is: {predicted_gpa[0]:.1f}")
```

## Predicted Output

```
The predicted GPA is: 4.0
```

## Contributing

Feel free to fork this repository, make improvements, and submit pull requests.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
