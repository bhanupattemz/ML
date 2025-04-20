from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
dataset = fetch_california_housing()
X = dataset.data
y = dataset.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = MLPRegressor(hidden_layer_sizes=(10,), activation='relu', max_iter=1000, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Scores
train_score = model.score(X_train, y_train)  # R^2 on train
test_score = model.score(X_test, y_test)     # R^2 on test

# Predictions & error
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Results
print(f"Train R² Score: {train_score:.4f}")
print(f"Test R² Score : {test_score:.4f}")
print(f"Mean Squared Error: {mse:.4f}")
