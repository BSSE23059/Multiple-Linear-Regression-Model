# Multiple-Linear-Regression-Model

This project implements a **Multiple Linear Regression** model from scratch in Python. It provides methods to fit the model to training data and make predictions on test data. The implementation highlights the mathematical foundations of regression and supports multiple input features.

---

## Overview

### What is Multiple Linear Regression?

Multiple Linear Regression models the relationship between a dependent variable $`Y`$ and multiple independent variables $`X_1, X_2, \dots, X_n`$:
$`
Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + \dots + \beta_nX_n + \epsilon
`$
Where:
- $Y$: Dependent variable (target)
- $X_i$: Independent variables (features)
- $\beta_0$: Intercept
- $\beta_i$: Coefficients for each feature
- $\epsilon$: Error term

The coefficients \(\beta\) are calculated using the **Normal Equation**:
$`
\beta = (X^T X)^{-1} X^T Y
`$

---

## Features

- Implements Multiple Linear Regression without external libraries like `scikit-learn`.
- Uses the **Normal Equation** for coefficient calculation.
- Provides methods to:
  - **Train** the model (`fit` method).
  - **Predict** outcomes for new data (`predict` method).

---

## Code Breakdown

### 1. **Initialization**
The `__init__` method initializes the model, setting placeholders for the coefficients and intercept.

```python
class MultipleLinearRegression:
    def __init__(self):
        self.coef_ = None  # Placeholder for coefficients (betas)
        self.intercept_ = None  # Placeholder for intercept (beta_0)
```

---

### 2. **Fit Method**
The `fit` method trains the model by calculating the coefficients and intercept using the **Normal Equation**:
$`
\beta = (X^T X)^{-1} X^T Y
`$

Steps:
1. **Augment the Features**: Adds a column of ones to $`X`$ to account for the intercept $\beta_0$.
2. **Compute Betas**:
   - Calculate $`(X^T X)^{-1}`$.
   - Multiply it by $`X^T Y`$ to find the coefficients.

```python
def fit(self, x_train, y_train_):
    # Add intercept term to features
    x_train = np.insert(x_train, 0, 1, axis=1)
    # Calculate coefficients using Normal Equation
    betas = np.linalg.inv(np.dot(x_train.T, x_train)).dot(x_train.T).dot(y_train_)
    self.intercept_ = betas[0]  # Intercept (beta_0)
    self.coef_ = betas[1:]  # Coefficients (beta_1, beta_2, ...)
```

---

### 3. **Predict Method**
The `predict` method uses the learned coefficients to make predictions:
$`
Y_{\text{pred}} = X \cdot \beta + \beta_0
`$

Steps:
1. Multiply the feature matrix $`X`$ by the coefficients.
2. Add the intercept term to calculate predictions.

```python
def predict(self, x_test):
    Y_pred = np.dot(x_test, self.coef_) + self.intercept_
    return Y_pred
```

---

## Example Usage

```python
import numpy as np
from multiple_linear_regression import MultipleLinearRegression

# Sample data
x_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y_train = np.array([3, 5, 7, 9])

# Initialize the model
model = MultipleLinearRegression()

# Train the model
model.fit(x_train, y_train)

# Print learned parameters
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Predict on new data
x_test = np.array([[5, 6], [6, 7]])
y_pred = model.predict(x_test)
print("Predictions:", y_pred)
```

---

## Mathematical Explanation

### 1. **Normal Equation**

The coefficients are calculated as:
$`
\beta = (X^T X)^{-1} X^T Y
`$

- $`X^T X`$: Dot product of the transposed feature matrix and the feature matrix.
- $`(X^T X)^{-1}`$: Inverse of the resulting matrix.
- $`X^T Y`$: Dot product of the transposed feature matrix and the target vector.

### 2. **Prediction Formula**

Predictions are made using:
$`
Y_{\text{pred}} = X \cdot \beta + \beta_0
`$

---

## Limitations

1. **Singular Matrix Error**:
   - If $`X^T X`$ is singular (non-invertible), the model will fail. This can happen due to:
     - Redundant features.
     - Insufficient data.

2. **Performance**:
   - The `fit` method relies on matrix inversion, which is computationally expensive for large datasets.

---

## Future Enhancements

- Add support for regularization (e.g., Ridge or Lasso Regression).
- Handle singular matrices using pseudo-inverse (`np.linalg.pinv`).
- Extend to handle categorical features.

---


