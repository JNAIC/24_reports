# task1 修正
## 手搓的线性回归模型
## 代码如下：
import numpy as np  
import matplotlib.pyplot as plt   
import pandas as pd    
url = "D:\作业.c\data.txt"    
data = pd.read_csv(url,names=['x','y'])  
x=data['x'].values    
y=data['y'].values  
#设置权重和偏置（初始值）  
w=0  
b=0  
#设置学习率  
learning_rate=0.01  
#设置epoch(轮数)  
epoch=1000  
#设置代价函数  
def cost_fuction(x,y,w,b)  
    m=len(y)  
    loss=(1/2*m)*np.sum(((w*x+b)-y)**2)  
    return loss  
#设置梯度下降  
def down(x,y,w,b,learning_rate):  
    m=len(y)  
    dw = 1/m*np.sum(((w*x+b)-y)*x)  
    db = 1/m*np.sum((w*x+b)-y)  
    w -= dw * learning_rate  
    b -= db * learning_rate  
    return w,b<br>
#训练模型<br>
for i in range(epoch):  
    w,b = down(x,y,w,b,learning_rate)  
    if epoch%100==0:  
        loss = cost_fuction(x,y,w,b)  
#可视化操作<br>
y1 = w * x + b  
plt.scatter(x,y,label='data point',color='blue')  
plt.plot(x,y1,label='predict line',color='red')  
plt.xlabel('population')  
plt.ylabel('profit')  
plt.legend()  
plt.show()  
#可视化图片  
![image](https://github.com/user-attachments/assets/586245a9-87ec-45df-bfed-ce40df4f0d39)  
我在之前做了一个套最小二乘法公式的线性回归  
图像如下：  
![image](https://github.com/user-attachments/assets/1d04a622-e42e-4092-b7ee-d7aebde8b78e)  
对比了一下，感觉还是有点差异，不过感觉拟合效果还可以。下次可以试试10000轮(会不会有点太过了0^0)。  





