from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

iris_data=load_iris()

X=iris_data.data
y=iris_data.target
target_names =iris_data.target_names

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

clf=KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

accuracy=accuracy_score(y_pred,y_test)

print(f"accuracy : {accuracy*100:.2f}%")

for i in range(len(y_pred)):
    predicted=target_names[y_pred[i]]
    actual=target_names[y_test[i]]
    comp="right" if y_pred[i]==y_test[i] else "wrong"
    print(f"No {i+1}. Predicted={predicted}, actual={actual}->{comp}")