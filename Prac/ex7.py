import csv
import math
import random

# Load dataset
data = []
header = []

with open(r"c:\Users\bhanu\OneDrive\Desktop\@jntua\ML_lab\Prac\fruit.csv", "r") as f:
    data = f.readlines()
    header = data[0].strip().split(",")
    data = [line.strip().split(",") for line in data[1:]]

# Shuffle data and split into training and testing sets (80% train, 20% test)
random.shuffle(data)
split_index = int(0.8 * len(data))
train_data = data[:split_index]
test_data = data[split_index:]

# Function to train Naïve Bayes model
def train_naive_bayes(data):
    y_value = sum(1 for row in data if row[-1] == "Yes")
    n_value = len(data) - y_value

    # Prior probabilities
    prob_yes = y_value / len(data)
    prob_no = n_value / len(data)

    # Conditional probabilities storage
    prob = {}

    for i in range(len(header) - 1):
        prob[header[i]] = {}
        values = {}

        for j in range(len(data)):
            if data[j][i] not in values:
                values[data[j][i]] = {"Yes": 1, "No": 1}  # Laplace smoothing

            if data[j][-1] == "Yes":
                values[data[j][i]]["Yes"] += 1
            else:
                values[data[j][i]]["No"] += 1

        for key, val in values.items():
            prob[header[i]][key] = {
                "Yes": val["Yes"] / (y_value + len(values)),  # Laplace smoothing
                "No": val["No"] / (n_value + len(values))
            }

    return prob, prob_yes, prob_no

# Function to classify using Naïve Bayes
def classify(prob, prob_yes, prob_no, input_data):
    log_prob_yes = math.log(prob_yes)
    log_prob_no = math.log(prob_no)

    for attr, val in input_data.items():
        if val in prob[attr]:
            log_prob_yes += math.log(prob[attr][val]["Yes"])
            log_prob_no += math.log(prob[attr][val]["No"])
        else:
            log_prob_yes += math.log(1 / (prob_yes + len(prob[attr])))
            log_prob_no += math.log(1 / (prob_no + len(prob[attr])))

    prob_yes = math.exp(log_prob_yes)
    prob_no = math.exp(log_prob_no)
    total = prob_yes + prob_no
    prob_yes /= total
    prob_no /= total

    return "Yes" if prob_yes > prob_no else "No"

# Train model
prob, prob_yes, prob_no = train_naive_bayes(train_data)

# Evaluate on test data
TP = TN = FP = FN = 0

for row in test_data:
    actual = row[-1]
    input_data = {header[i]: row[i] for i in range(len(header) - 1)}
    predicted = classify(prob, prob_yes, prob_no, input_data)

    if predicted == "Yes" and actual == "Yes":
        TP += 1
    elif predicted == "No" and actual == "No":
        TN += 1
    elif predicted == "Yes" and actual == "No":
        FP += 1
    elif predicted == "No" and actual == "Yes":
        FN += 1

# Compute metrics
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) != 0 else 0
recall = TP / (TP + FN) if (TP + FN) != 0 else 0

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
