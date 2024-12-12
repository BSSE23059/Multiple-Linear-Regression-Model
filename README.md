# Multiple-Linear-Regression-Model
Building Linear Regression class model for Multiple Linear Regression.
This Python project implements a Multiple Linear Regression model from scratch, providing methods to fit the model to training data and make predictions on test data. It demonstrates the mathematical foundations of regression while allowing users to work with multiple input features.

Overview
What is Multiple Linear Regression?
Multiple Linear Regression is a statistical technique that models the relationship between a dependent variable 
𝑌
Y and multiple independent variables 
𝑋
1
,
𝑋
2
,
…
,
𝑋
𝑛
X 
1
​
 ,X 
2
​
 ,…,X 
n
​
 . The relationship is expressed as:

𝑌
=
𝛽
0
+
𝛽
1
𝑋
1
+
𝛽
2
𝑋
2
+
⋯
+
𝛽
𝑛
𝑋
𝑛
+
𝜖
Y=β 
0
​
 +β 
1
​
 X 
1
​
 +β 
2
​
 X 
2
​
 +⋯+β 
n
​
 X 
n
​
 +ϵ
Where:

𝑌
Y: Dependent variable (target)
𝑋
𝑖
X 
i
​
 : Independent variables (features)
𝛽
0
β 
0
​
 : Intercept
𝛽
𝑖
β 
i
​
 : Coefficients for each feature
𝜖
ϵ: Error term
This project calculates the coefficients (
𝛽
β) using the Normal Equation:

𝛽
=
(
𝑋
𝑇
𝑋
)
−
1
𝑋
𝑇
𝑌
β=(X 
T
 X) 
−1
 X 
T
 Y
Features
Implements Multiple Linear Regression without relying on external libraries like scikit-learn.
Uses the Normal Equation to calculate coefficients.
Provides methods to:
Train the model (fit method).
Predict outcomes for new data (predict method).
Code Breakdown
1. Initialization
The __init__ method initializes the model, setting placeholders for the coefficients and intercept.

python
Copy code
def __init__(self):
    self.coef_ = None  # Placeholder for coefficients (betas)
    self.intercept_ = None  # Placeholder for intercept (beta_0)
2. Fit Method
The fit method trains the model by calculating the coefficients and intercept using the Normal Equation:

𝛽
=
(
𝑋
𝑇
𝑋
)
−
1
𝑋
𝑇
𝑌
β=(X 
T
 X) 
−1
 X 
T
 Y
Steps:

Augment the Features: Adds a column of ones to 
𝑋
X to account for the intercept (
𝛽
0
β 
0
​
 ).
Compute Betas:
Calculate 
(
𝑋
𝑇
𝑋
)
−
1
(X 
T
 X) 
−1
 .
Multiply it by 
𝑋
𝑇
𝑌
X 
T
 Y to find the coefficients.
python
Copy code
def fit(self, x_train, y_train_):
    # Add intercept term to features
    x_train = np.insert(x_train, 0, 1, axis=1)
    # Calculate coefficients using Normal Equation
    betas = np.linalg.inv(np.dot(x_train.T, x_train)).dot(x_train.T).dot(y_train_)
    self.intercept_ = betas[0]  # Intercept (beta_0)
    self.coef_ = betas[1:]  # Coefficients (beta_1, beta_2, ...)
3. Predict Method
The predict method uses the learned coefficients to make predictions:

𝑌
pred
=
𝑋
⋅
𝛽
+
𝛽
0
Y 
pred
​
 =X⋅β+β 
0
​
 
Steps:

Multiply the feature matrix 
𝑋
X by the coefficients.
Add the intercept term to calculate predictions.
python
Copy code
def predict(self, x_test):
    Y_pred = np.dot(x_test, self.coef_) + self.intercept_
    return Y_pred
Example Usage
python
Copy code
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
Mathematical Explanation
Normal Equation:

Coefficients are calculated using:
𝛽
=
(
𝑋
𝑇
𝑋
)
−
1
𝑋
𝑇
𝑌
β=(X 
T
 X) 
−1
 X 
T
 Y
The intercept is the first value of 
𝛽
β (
𝛽
0
β 
0
​
 ), and the rest are the coefficients.
Prediction Formula:

Predictions are made using:
𝑌
pred
=
𝑋
⋅
𝛽
+
𝛽
0
Y 
pred
​
 =X⋅β+β 
0
​
 
Matrix Operations:

𝑋
𝑇
𝑋
X 
T
 X: Computes the dot product of the transposed feature matrix and the feature matrix.
(
𝑋
𝑇
𝑋
)
−
1
(X 
T
 X) 
−1
 : Inverts the resulting matrix.
𝑋
𝑇
𝑌
X 
T
 Y: Computes the dot product of the transposed feature matrix and the target vector.
Limitations
Singular Matrix Error:

If 
𝑋
𝑇
𝑋
X 
T
 X is singular (non-invertible), the model will fail. This can happen due to:
Redundant features.
Insufficient data.
Performance:

The fit method relies on matrix inversion, which is computationally expensive for large datasets.
Future Enhancements
Add support for regularization (e.g., Ridge or Lasso Regression).
Handle singular matrices using pseudo-inverse (np.linalg.pinv).
Extend to handle categorical features.
