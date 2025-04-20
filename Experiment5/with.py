import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

iris = pd.read_csv(r"c:\Users\bhanu\OneDrive\Desktop\@jntua\ML_lab\Experiment4\iris.csv")
X = iris.iloc[:, :-1].values
y = iris.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

y_pred = nb_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(report)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=nb_classifier.classes_, 
            yticklabels=nb_classifier.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

feature_importance = np.abs(nb_classifier.theta_)
feature_names = iris.columns[:-1]

plt.figure(figsize=(10, 6))
for i, class_name in enumerate(nb_classifier.classes_):
    plt.bar(feature_names, feature_importance[i], alpha=0.7, label=class_name)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance per Class')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()