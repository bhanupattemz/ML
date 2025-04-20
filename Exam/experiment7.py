from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import pandas as pd

label_encoder=LabelEncoder()
data=pd.read_csv(r"c:/Users/bhanu/OneDrive/Desktop/@jntua/ML_lab/Exam/heart.csv")
for column in data.columns:
    data[column]=label_encoder.fit_transform(data[column])

X=data.iloc[:,:-1]
y=data.iloc[:,-1]
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.2,random_state=42)

model=GaussianNB()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
accuracy=accuracy_score(y_pred,y_test)

print("accuracy :",accuracy*100)