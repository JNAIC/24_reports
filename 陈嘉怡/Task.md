* 最小二乘法
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
pd.read_csv('data.txt', header=None) 
x = data.iloc[:, 0]
y = data.iloc[:, 1]

# 最小二乘法公式
X = np.c_[np.ones((len(x),1)),x]
beta=np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y.values)

# 图像展示
plt.scatter(x,y)
plt.plot(x,beta[0]+beta[1]*x,c='black')
plt.show()
```
* 可视化
  
  ![image](https://github.com/user-attachments/assets/e672cc46-c334-4306-8d56-28771ea31146)

* 使用梯度下降
```python
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
data = pd.read_csv('data.txt', header=None) 
x = data.iloc[:, 0]
y = data.iloc[:, 1]

# 损失函数
def computer_cost(x,y,w,b):
    m=x.shape[0]
    cost=0
    for i in range(m):
        f_wb=w*x[i]+b
        cost=cost+(f_wb-y[i])**2
    total_cost=1/(2*m)*cost
    return total_cost

# 偏导
def computer_gradient(x,y,w,b):
    m=x.shape[0]
    dj_dw=0
    dj_db=0
    for i in range(m):
        f_wb=w*x[i]+b
        dj_dw_i=(f_wb-y[i])*x[i]
        dj_db_i=f_wb-y[i]
        dj_dw+=dj_dw_i
        dj_db+=dj_db_i
    dj_dw=dj_dw/m
    dj_db=dj_db/m
    return dj_dw,dj_db

# 梯度下降
def gradient_descent(x,y,w_in,b_in,alpha,num_iters,cost_function,gradient_function):
    b=b_in
    w=w_in
    for i in range(num_iters):
        dj_dw,dj_db=gradient_function(x,y,w,b)
        b=b-alpha*dj_db
        w=w-alpha*dj_dw
    return w,b

# 设置w，b的初始值均为1
w_init=0
b_init=0

iterations=10000
tmp_alpha=1.0e-2
w_final,b_final=gradient_descent(x,y,w_init,b_init,tmp_alpha,
                                 iterations, computer_cost, computer_gradient)
print(f"w={w_final}")
print(f"b={b_final}")

# 可视化
plt.scatter(x,y)
plt.plot(x,b_final+w_final*x,c='black')
plt.show()
```
* 可视化结果
  
  ![image](https://github.com/user-attachments/assets/771d8cd4-2bc5-448f-b7ce-f908df648470)
  
![image](https://github.com/user-attachments/assets/d64b16b8-d350-4d72-b326-0f6e4975b5dd)
