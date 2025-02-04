```
import numpy as np
import pandas as pd
train_data = pd.read_csv(r'C:\Code\train.csv')
test_data = pd.read_csv(r'C:\Code\test.csv')

#补充缺失值
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].median())

#数据整理
train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})
test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})
features = ['Sex','Pclass','Fare','Age','SibSp','Parch']
X_train = train_data[features].values
y_train = train_data['Survived'].values
X_test = test_data[features].values

#logistic 回归
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def costfunction(X, y, theta,lambda_):
    m = len(y)
    h = sigmoid(X @ theta)
    loss = (-1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    reg_loss=(lambda_/(2*m))*np.sum(theta[1:]**2)
    return loss+reg_loss

def gradient_descent(X, y, theta, learning_rate, num_iterations,lambda_):
    m = len(y)
    for _ in range(num_iterations):
        h = sigmoid(X @ theta)
        gradient = (1/m) * (X.T @ (h - y))
        gradient[1:]+=(lambda_/m)*theta[1:]
        theta -= learning_rate * gradient
    return theta

X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
theta = np.zeros(X_train.shape[1])
learning_rate = 0.01
num_iterations = 50000
lambda_=0.1
theta = gradient_descent(X_train, y_train, theta, learning_rate, num_iterations,lambda_)

probabilities = sigmoid(X_test @ theta)
predictions = (probabilities >= 0.5).astype(int)
output = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': predictions})
output.to_csv('submission.csv', index=False)
```
