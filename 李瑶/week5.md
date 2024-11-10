# 进击的ai人2.0
## python基础应该是跑的差不多了，我原本想跑一下minst数据集的，结果好像在导入数据集那出差错了，我下周再研究一下，不过下周应该就进入训练模型的学习了，这周感觉效率一般，最后还是交一些代码吧。
import numpy as np
import matplotlib.pyplot as plt
def relu(x):
    return np.maximum(0,x)
x=np.arange(-6,6,0.1)
y=relu(x)
plt.plot(x,y,label='relu')
plt.legend()
plt.show()
![image](https://github.com/user-attachments/assets/08b0190b-1875-46d3-bdb1-e55e8031ae79)
## 上面是relu函数的图像。
## 不足：感觉效率还是太慢了，我会鞭策我自己的。加油吧！
