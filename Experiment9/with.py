from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train k-NN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predict
y_pred = knn.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%\n")

# Print correct and incorrect predictions
print("Prediction Results:")
for i in range(len(y_test)):
    predicted = target_names[y_pred[i]]
    actual = target_names[y_test[i]]
    status = "Correct" if y_pred[i] == y_test[i] else "Wrong"
    print(f"Sample {i+1}: Predicted = {predicted}, Actual = {actual} -> {status}")
