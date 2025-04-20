from sklearn.tree import DecisionTreeClassifier,export_text
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

data=pd.read_csv( r"c:\Users\bhanu\OneDrive\Desktop\@jntua\ML_lab\Experiment3\games.csv")

label_encoder=LabelEncoder()
for column in data.columns:
    data[column]=label_encoder.fit_transform(data[column])

X=data.iloc[:,:-1]
y=data.iloc[:,-1]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

clf=DecisionTreeClassifier(criterion="entropy",random_state=42)
clf.fit(X_train,y_train)

tree_rules=export_text(clf,feature_names=list(X.columns))
print("Decision Tree :")
print(tree_rules)

y_pred=clf.predict(X_test)
a=accuracy_score(y_pred,y_test)

print("accuracy_score:",a*100)