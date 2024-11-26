* 代码
```python
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv('advertising.csv', header=0) 
x=data[['wechat','weibo','others']]
y=data['sales']

# 定义数据归一化函数
def normalize(x):
    min_x = x.min(axis=0)
    max_x = x.max(axis=0)
    x_norm = (x - min_x) / (max_x - min_x)
    return x_norm, min_x, max_x
# x，y归一
x_norm,min_x,max_x=normalize(x)
y_norm,min_y,max_y=normalize(y)

# 定义损失函数
def compute_cost(x,y,w,b):
    m=x.shape[0]
    cost=0
    for i in range(m):
        f_wb_i=np.dot(x.iloc[i],w)+b
        cost=cost+(f_wb_i-y.iloc[i])**2
    cost=1/(2*m)*cost
    return cost

# 定义梯度函数
def compute_gradient(x,y,w,b):
    m,n=x.shape
    dj_dw=np.zeros((n,))
    dj_db=0
    for i in range(m):
        err=(np.dot(x.iloc[i],w)+b)-y.iloc[i]
        for j in range(n):
            dj_dw[j]=dj_dw[j]+err*x.iloc[i,j]
        dj_db=dj_db+err
    dj_dw=dj_dw/m
    dj_db=dj_db/m
    return dj_dw,dj_db

# 定义梯度下降函数
def gradient_descent(x,y,w_in,b_in,cost_function,gradient_function,alpha,num_iters):
    b=b_in
    w=w_in
    cost_history = []
    for i in range(num_iters):
        dj_dw,dj_db=gradient_function(x,y,w,b)
        b=b-alpha*dj_db
        w=w-alpha*dj_dw
        cost = cost_function(x, y, w, b)
        cost_history.append(cost)
    return w,b,cost_history

# 设置w和b的初始值，梯度下降次数和学习率
initial_w=np.zeros(x.shape[1])
initial_b=0
iterations=1000
tmp_alpha=0.01
# 梯度下降，得出结果
w_final,b_final,cost_history=gradient_descent(x_norm,y_norm,initial_w,initial_b, compute_cost, compute_gradient,tmp_alpha,
                                 iterations)
print(f"w={w_final}")
print(f"b={b_final}")
def predict(x, w, b):
    return np.dot(x, w) + b
 
# 绘制代价函数变化
plt.plot(range(iterations), cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function Over Iterations')
plt.show()
```
* 代价函数可视化
  
![image](https://github.com/user-attachments/assets/afba062e-e0f8-42e7-b267-e062ae8ea3bb)

* 运行结果可视化
  
![image](https://github.com/user-attachments/assets/4cc135c1-81d5-412d-9c33-a8ec2cfb119a)

* 收获  
  学习了数据数据归一化的基本操作  
  知道了多变量线性回归中归一化操作是为了消除量纲的影响
