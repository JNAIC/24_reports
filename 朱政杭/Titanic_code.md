```python
import numpy as np
import pandas as pd
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
#填充缺失值
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].median())
train_data['Fare'] = train_data['Fare'].fillna(train_data['Fare'].median())
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].median())
train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])
test_data['Embarked'] = test_data['Embarked'].fillna(test_data['Embarked'].mode()[0])
#数据处理
train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})
test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})
train_data['Embarked'] = train_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
test_data['Embarked'] = test_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
train_data['Title'] = train_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
test_data['Title'] = test_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
title_mapping = {
    'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3,
    'Dr': 4, 'Rev': 4, 'Col': 4, 'Major': 4, 'Sir': 4, 'Don': 4, 'Jonkheer': 4,
    'Lady': 5, 'Countess': 5, 'Mme': 5, 'Ms': 5, 'Mlle': 5}
train_data['Title'] = train_data['Title'].map(title_mapping).fillna(0)
test_data['Title'] = test_data['Title'].map(title_mapping).fillna(0)
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1
train_data['IsAlone'] = (train_data['FamilySize'] == 1).astype(int)
test_data['IsAlone'] = (test_data['FamilySize'] == 1).astype(int)
# 选择特征
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
            'Embarked','Title','FamilySize','IsAlone']
X_train = train_data[features].values
y_train = train_data['Survived'].values
X_test = test_data[features].values
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def costfunction(X, y, theta,lambda_):
    m = len(y)
    h = sigmoid(X @ theta)
    loss = (-1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    reg_loss=(lambda_/(2*m))*np.sum(theta[1:]**2)#L2正则化
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
learning_rate = 0.001
num_iterations = 2000000
lambda_=0.1
theta = gradient_descent(X_train, y_train, theta, learning_rate, num_iterations,lambda_)
probabilities = sigmoid(X_test @ theta)
predictions = (probabilities >= 0.5).astype(int)
output = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': predictions})
output.to_csv('submission3.csv', index=False)
```
