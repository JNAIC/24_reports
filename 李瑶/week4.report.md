## 进击的ai人
# 这周的学习内容：
# 1、把类的知识系统地梳理了一遍
# 2、提前了解了矩阵的相乘，利用numpy进行了一个简单的神经网络搭建，不过还没有涉及到代价函数设置，还有后向的神经网络调参正在进行中。以下是一个简单的三层神经网络搭建

import numpy as np
def sigmod(x):
    return 1/(1+np.exp(-x))
def init_network():
    nw={}
    nw['w1']=np.array([[0.1,0.2,0.3],[0.2,0.3,0.4]])
    nw['b1']=np.array([0.1,0.2,0.3])
    nw['w2']=np.array([[0.1,0.2],[0.2,0.3],[0.3,0.4]])
    nw['b2']=np.array([0.1,0.2])
    nw['w3']=np.array([[0.2,0.3],[0.1,0.4]])
    nw['b3']=np.array([0.4,0.5])
    return nw
def network(x):
    net=init_network()
    A1=np.dot(x,net['w1'])+net['b1']
    Z1=sigmod(A1)
    A2=np.dot(Z1,net['w2'])+net['b2']
    Z2=sigmod(A2)
    A3=np.dot(Z2,net['w3'])+net['b3']
    y=A3
    return y
x=np.array([2,3])
y=network(x)
print(y)
# 不足：我的时间太少了，不过我会努力平衡好时间的。还有就是我的学习路线和学长们规划的有点出路，我会努力中和的，不过我先把我之前准备好的计划先执行了，之后会跟上pandas的学习的。
