import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Load dataset
file_path = r"c:\Users\bhanu\OneDrive\Desktop\@jntua\ML_lab\Exam\heart.csv"
df = pd.read_csv(file_path)

# Select features for regression (age -> chol)
X = df[['age']].values
y = df['chol'].values

# LWR function
def locally_weighted_regression(X_train, y_train, X_test, tau=10):
    y_pred = np.zeros(len(X_test))
    for i, x in enumerate(X_test):
        W = np.exp(-cdist(X_train, [x], 'sqeuclidean') / (2 * tau ** 2))
        W = np.diag(W.flatten())
        theta = np.linalg.pinv(X_train.T @ W @ X_train) @ X_train.T @ W @ y_train
        y_pred[i] = np.dot(x, theta)
    return y_pred

# Generate test inputs
X_test = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_pred = locally_weighted_regression(X, y, X_test, tau=10)

# Plot
plt.scatter(X, y, color='red', label="Actual Data")
plt.plot(X_test, y_pred, color='blue', label="LWR Prediction")
plt.xlabel("Age")
plt.ylabel("Cholesterol")
plt.title("Locally Weighted Regression (Age vs Chol)")
plt.legend()
plt.grid(True)
plt.show()
