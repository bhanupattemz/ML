from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error



dataset=fetch_california_housing()
X=dataset.data
y=dataset.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=MLPRegressor(hidden_layer_sizes=(10,),activation="relu",max_iter=1000,random_state=42)

model.fit(X_train,y_train)
train_score=model.score(X_train,y_train)
test_score=model.score(X_test,y_test)

y_pred=model.predict(X_test)
error=mean_squared_error(y_pred,y_test)

print("train score :",train_score)
print("test Score :",test_score)
print("Mean sq error :", error)