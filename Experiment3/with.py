from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd

filename = r"c:\Users\bhanu\OneDrive\Desktop\@jntua\ML_lab\Experiment3\games.csv"
data = pd.read_csv(filename)

label_encoder = LabelEncoder()

for column in data.columns:
    data[column] = label_encoder.fit_transform(data[column])

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X_train, y_train)

tree_rules = export_text(clf, feature_names=list(X.columns))
print("Decision Tree Rules:")
print(tree_rules)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
