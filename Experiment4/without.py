import numpy as np
import csv
import random
import math

def load_data(filename):
    data = []
    labels = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) > 0:
                features = [float(x) for x in row[:-1]]
                data.append(features)
                label = row[-1]
                if label == "Iris-setosa":
                    labels.append([1, 0, 0])
                elif label == "Iris-versicolor":
                    labels.append([0, 1, 0])
                else:
                    labels.append([0, 0, 1])
    return np.array(data), np.array(labels)

def normalize(X):
    X_norm = np.zeros_like(X, dtype=float)
    for i in range(X.shape[1]):
        col_min = np.min(X[:, i])
        col_max = np.max(X[:, i])
        X_norm[:, i] = (X[:, i] - col_min) / (col_max - col_min)
    return X_norm

def train_test_split(X, y, test_size=0.2):
    combined = list(zip(X, y))
    random.shuffle(combined)
    X, y = zip(*combined)
    X, y = np.array(X), np.array(y)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    return X_train, X_test, y_train, y_test

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.b2 = np.zeros((1, self.output_size))
        self.loss_history = []
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, output):
        m = X.shape[0]
        self.output_error = y - output
        self.output_delta = self.output_error * self.sigmoid_derivative(output)
        self.hidden_error = np.dot(self.output_delta, self.W2.T)
        self.hidden_delta = self.hidden_error * self.sigmoid_derivative(self.a1)
        self.W2 += np.dot(self.a1.T, self.output_delta) * self.learning_rate
        self.b2 += np.sum(self.output_delta, axis=0, keepdims=True) * self.learning_rate
        self.W1 += np.dot(X.T, self.hidden_delta) * self.learning_rate
        self.b1 += np.sum(self.hidden_delta, axis=0, keepdims=True) * self.learning_rate
    
    def compute_loss(self, y_true, y_pred):
        return np.mean(np.sum(np.square(y_true - y_pred), axis=1))
    
    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            output = self.forward(X)
            loss = self.compute_loss(y, output)
            self.loss_history.append(loss)
            self.backward(X, y, output)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
    
    def predict(self, X):
        return self.forward(X)
    
    def evaluate(self, X, y):
        predictions = self.predict(X)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y, axis=1)
        accuracy = np.mean(predicted_classes == true_classes)
        return accuracy

if __name__ == "__main__":
    X, y = load_data( r"c:\Users\bhanu\OneDrive\Desktop\@jntua\ML_lab\Experiment4\iris.csv")
    X = normalize(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    input_size = X_train.shape[1]
    hidden_size = 8
    output_size = y_train.shape[1]
    nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate=0.01)
    nn.train(X_train, y_train, epochs=1000)
    train_accuracy = nn.evaluate(X_train, y_train)
    test_accuracy = nn.evaluate(X_test, y_test)
    print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")
