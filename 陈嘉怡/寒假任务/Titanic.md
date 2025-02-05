在拿到数据集后，可以看到Cabin缺失数量较多，姓名对结果影响不大，所以我选择将他们丢弃，然后对性别和港口用one-hot方法表示

我主要使用的是sklearn中的逻辑回归和随机森林来分别进行预测
```
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
data0=pd.read_csv('D:\\VS code project\\Titanic\\train.csv')
x=pd.get_dummies(data0['Sex'],prefix='Sex',dtype=int)
data0=pd.concat([data0,x],axis=1)
x=pd.get_dummies(data0['Embarked'],prefix='Embarked',dtype=int)
data0=pd.concat([data0,x],axis=1)
y=data0['Survived']
data0=data0.drop(columns=['Name','Sex','Ticket','Cabin','Embarked','Survived'])
clear_data=data0.fillna(data0.mean())

x=clear_data
x_train,x_test,y_train,y_test=train_test_split(x,y,stratify=y,random_state=42)
lr=LogisticRegression().fit(x_train,y_train)
print(lr.score(x_train, y_train))
print(lr.score(x_test, y_test))
rf=RandomForestClassifier().fit(x_train,y_train)
test=pd.read_csv('D:\\VS code project\\Titanic\\test.csv')
x=pd.get_dummies(test['Sex'],prefix='Sex',dtype=int)
test=pd.concat([test,x],axis=1)
x=pd.get_dummies(test['Embarked'],prefix='Embarked',dtype=int)
test=pd.concat([test,x],axis=1)
test=test.drop(columns=['Name','Sex','Ticket','Cabin','Embarked'])
test=test.fillna(test.mean())
print(rf.score(x_train, y_train))
print(rf.score(x_test, y_test))
y_pred=lr.predict(test[:])
y_pred2=rf.predict(test[:])
x_test=test['PassengerId'].values
result=pd.DataFrame({'PassengerId':x_test,'Survived':y_pred})
result2=pd.DataFrame({'PassengerId':x_test,'Survived':y_pred2})
result.to_csv('D:\\VS code project\\Titanic\\result0.csv',index=False)
result2.to_csv('D:\\VS code project\\Titanic\\result1.csv',index=False)
```
训练集和测试集得到的分数分别为：

![屏幕截图 2025-01-24 163719](https://github.com/user-attachments/assets/0d860816-3de1-42b7-807a-8a2c61a8500b)

最终提交的结果为：
![image](https://github.com/user-attachments/assets/323cde01-bf4a-40ef-a401-8bbaa605f1b4)

我还想继续提高正确率，我想着是否可以不将训练集划分为训练集和测试集，这样就减少了自我检测的部分，但是可以增大训练集的数量
```
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
data0=pd.read_csv('D:\\VS code project\\Titanic\\train.csv')
x=pd.get_dummies(data0['Sex'],prefix='Sex',dtype=int)
data0=pd.concat([data0,x],axis=1)
x=pd.get_dummies(data0['Embarked'],prefix='Embarked',dtype=int)
data0=pd.concat([data0,x],axis=1)
y=data0['Survived']
data0=data0.drop(columns=['Name','Sex','Ticket','Cabin','Embarked','Survived'])
clear_data=data0.fillna(data0.mean())

x=clear_data
lr=LogisticRegression().fit(x,y)
rf=RandomForestClassifier().fit(x,y)
test=pd.read_csv('D:\\VS code project\\Titanic\\test.csv')
x=pd.get_dummies(test['Sex'],prefix='Sex',dtype=int)
test=pd.concat([test,x],axis=1)
x=pd.get_dummies(test['Embarked'],prefix='Embarked',dtype=int)
test=pd.concat([test,x],axis=1)
test=test.drop(columns=['Name','Sex','Ticket','Cabin','Embarked'])
test=test.fillna(test.mean())
y_pred=lr.predict(test[:])
y_pred2=rf.predict(test[:])
x_test=test['PassengerId'].values
result=pd.DataFrame({'PassengerId':x_test,'Survived':y_pred})
result2=pd.DataFrame({'PassengerId':x_test,'Survived':y_pred2})
result.to_csv('D:\\VS code project\\Titanic\\answer0.csv',index=False)
result2.to_csv('D:\\VS code project\\Titanic\\answer1.csv',index=False)
```
![image](https://github.com/user-attachments/assets/2c4fab4b-a3c9-46ff-b503-4b4a7ad618bc)
从最后得出的结果来看，得分确实略有提升
