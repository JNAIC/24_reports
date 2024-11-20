```python
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
data=pd.read_csv('data.txt',header=None,names=['population','profit'])
data.insert(0,'ones',1)
x=data.iloc[:,0:2]
y=data.iloc[:,-1]
x=np.array(x)
y=np.array(y)
y=y.reshape(97,1)
def costFunction(x,y,theta):
    inner=np.power(x.dot(theta)-y,2)
    return np.sum(inner)/2*len(x)
def gradientDescent(x,y,theta,alpha,iters):
    costs=[]
    for i in range(iters):
        theta=theta-x.T@(x@theta-y)*alpha/len(x)
        cost=costFunction(x,y, theta)
        costs.append(cost)
    return theta,costs
alpha=0.02
iters=2000
theta=np.zeros((2,1))
theta,costs=gradientDescent(x,y,theta,alpha,iters)
x_=np.linspace(y.min(),y.max(),100)
y_=theta[0,0]+theta[1,0]*x_
plt.scatter(x[:,1],y,label='trainning data',c='b',s=16)
plt.plot(x_,y_,'r',label='predict')
plt.legend()
plt.show()
```
### 可视化
![屏幕截图 2024-11-20 230243](https://github.com/user-attachments/assets/d7fc7fb2-03dd-4bcd-93c5-b06873f88f37)

