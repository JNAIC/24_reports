# TASK2
## 多变量线性回归模型
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
url="D:\\作业.c\\advertising.csv"  
data=pd.read_csv(url,names=["x1","x2","x3","y"])  
df=data[1:].astype(float)  
def min_max(df):  
    dt = df.copy()  
    for column in df.columns:  
        min_val = df[column].min()  
        max_val = df[column].max()  
        dt[column] = (df[column] - min_val) / (max_val - min_val)  
    return dt  
dt=min_max(df)  
print(dt.head())  
x=dt[["x1","x2","x3"]].values.astype(float)  
y=dt["y"].values.astype(float)  
print(x.shape)  
print(y.shape)  
w=np.array([0.1,0.1,0.1])  
b=1  
learning_rate = 0.01  
#设置梯度下降  
def down(x,y,w,b,learning_rate):  
    pre=x.dot(w)+b  
    m=len(y)  
    dw=(1/m)*np.sum(np.dot(x.T,(pre-y)))  
    db=(1/m)*np.sum(pre-y)  
    w-=learning_rate*dw  
    b-=learning_rate*db  
    return w,b  
epoch=100000  
for i in range(epoch):  
    w,b=down(x,y,w,b,learning_rate)  
print(w)  
print(b)  
y1=x.dot(w)+b  
x=np.mean(x,axis=1)  
plt.scatter(x,y,label="data point",color="blue")  
plt.plot(x,y1,label="predict line",color="red")  
plt.xlabel("average x")  
plt.ylabel("y")  
plt.legend()  
plt.show()  
## 可视化图片
![image](https://github.com/user-attachments/assets/e0ba5221-1e5a-4492-81b0-d6892f4276bd)
## 小唠嗑
啊啊啊，我一直卡在为什么x.dot(w)+b，我不理解为什么对应维度一样，但是就是会报错，哪来的字符串啊！？后来才知道pandas读取的数据是字符串，于是我学会了astype转换类型
然后我又卡在了为什么w,b是nan啊！！！后来我学会了min_max归一化，让数据在[0,1]之间，min_max应该是最简单的归一化了，总之，收获挺多的，嘻嘻，开心^0^!

