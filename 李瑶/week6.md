# 任务汇报
## 最开始我以为是套公式（最小二乘法）来达到预测目的，于是我写了如下代码
## import pandas as pd
## import matplotlib.pyplot as plt
## import numpy as np
## url="D:\作业.c\data.txt"
## # 读取数据
## data = pd.read_csv(url, names=['x', 'y'])
## # 提取x和y的值
## x = data['x'].values#x，y是一个列表（数组），分别储存着x和y对应的那一列数据。
## y = data['y'].values
## b = np.sum((x-np.mean(x))*(y-np.mean(y)))/np.sum((x-np.mean(x))*(x-np.mean(x)))
## a = np.mean(y)-b*np.mean(x)
## y1 = b * x + a
## plt.scatter(x,y,label='data point',color='blue')
## plt.plot(x,y1,label='predicted profit',color='red')
## plt.xlabel("population")
## plt.ylabel("profit")
## plt.legend()
## plt.show()
![image](https://github.com/user-attachments/assets/dffb757a-3b80-476c-813f-a6d5fe08f21b)

## 后来知道是机器学习，就是用训练模型的方法，然后我又写了如下代码
## import numpy as np
## import matplotlib.pyplot as plt
## import pandas as pd
## url = "D:\作业.c\data.txt"
## data = pd.read_csv(url,names=['x','y'])
## x=data['x'].values
## y=data['y'].values
## #设置权重和偏置（初始值）
## w=0
## b=0
## #设置学习率
## learning_rate=0.01
## #设置epoch(轮数)
## epoch=1000
## #设置代价函数

## def cost_fuction(x,y,w,b):
##    m=len(y)
##    loss=(1/2*m)*np.sum(((w*x+b)-y)**2)
##    return loss

## #设置梯度下降
## def down(x,y,w,b,learning_rate):
##    m=len(y)
##    dw = 1/m*np.sum(((w*x+b)-y)*x)
##    db = 1/m*np.sum((w*x+b)-y)
##    w -= dw * learning_rate
##    b -= db * learning_rate
##    return w,b
## #训练模型
## for i in range(epoch):
##     w,b = down(x,y,w,b,learning_rate)
##     if epoch%100==0:
##        loss = cost_fuction(x,y,w,b)
## #可视化操作
## y1 = w * x + b
## plt.scatter(x,y,label='data point',color='blue')
## plt.plot(x,y1,label='predict line',color='red')
## plt.xlabel('population')
## plt.ylabel('profit')
## plt.legend()
## plt.show()
![image](https://github.com/user-attachments/assets/2a6baef6-55bf-4ce6-9679-7c30f6698620)
## 感觉图像拟合的还可以，而且跑1000轮就可以了，这周太忙了，没有进一步学习神经网络，就是完成了下前辈交代的任务

