## 预测在未来的某个节点，一个特定的广告投放金额对应能实现的商品销售额
* Code
```python
import numpy as np
import matplotlib.pyplot as plt

# 读取数据文件
data = np.loadtxt('advertising.csv', delimiter=',')

# 提取数据
X = data[:, :-1]
y_true = data[:, -1]

# 在X的最前面加上一列全为1的列（偏置项bias）
X = np.hstack((np.ones((X.shape[0], 1)), X))  

# 特征值标准化
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
# 由于引入了偏置项，该列数据标准差为0，做除前改为1
X_std[X_std == 0] = 1  
X = (X - X_mean) / X_std  

# 真实值的标准化
y_mean = np.mean(y_true)  
y_std = np.std(y_true)  
y_true = (y_true - y_mean) / y_std

def lostfunction(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# 权重随机初始化
num_features = X.shape[1] 
weights = np.random.rand(num_features, 1) * 0.01 # 防止梯度爆炸
# 梯度下降
learning_rate = 0.01  
iterations = 2000  

for i in range(iterations):
  
    predictions = np.dot(X, weights)  
 
    lost = lostfunction(y_true.reshape(-1, 1), predictions)
    if i % 100 == 0:
        print(f"Iteration {i}: lost = {lost}")

    gradients_weights = (2 / X.shape[0]) * np.dot(X.T, (predictions - y_true.reshape(-1, 1)))

    # 更新模型参数
    weights -= learning_rate * gradients_weights

# 可视化功能
plt.figure
# 基于样本特征数据，得到预测值
final_predictions = np.dot(X, weights)
# 反标准化
final_predictions = final_predictions * y_std + y_mean  
# 绘制真实值
plt.scatter(range(len(y_true)), y_true * y_std + y_mean, label='True',color='green')
# 绘制预测值
plt.scatter(range(len(predictions)), final_predictions,  label='Pred',color='blue')    

# 添加图表标题和标签
plt.title('Fit condition')
plt.xlabel('sample number')
plt.ylabel('value')
plt.legend()
plt.grid(color='grey')
plt.show()

# 预测功能
def predict(weights, X_mean, X_std):
    x1 = float(input("请输入wechat投放量: "))
    x2 = float(input("请输入weibo投放量: "))
    x3 = float(input("请输入others投放量: "))
    
    # 将输入值标准化
    x1_standardized = (x1 - X_mean[1]) / X_std[1]
    x2_standardized = (x2 - X_mean[2]) / X_std[2]
    x3_standardized = (x3 - X_mean[3]) / X_std[3]
    
    # 构建输入特征矩阵并增加偏置项
    input_X = np.matrix([[1, x1_standardized, x2_standardized, x3_standardized]])  
    # 进行预测（预测输出已经标准化）
    prediction = np.dot(input_X, weights)
    
    # 反标准化预测值
    prediction = prediction * y_std + y_mean  # 逆标准化
    print(f"预测的收益值为: {prediction[0, 0]}")

# 使用训练后的权重进行预测
predict(weights, X_mean, X_std)

```
* 想法

  1.基于
  ![线性回归2](https://github.com/user-attachments/assets/589747da-ae34-4dd1-9ee3-9adc8eacbd8b)
  ![梯度计算公式](https://github.com/user-attachments/assets/20c859ed-7417-4934-8689-ca7091be5f8d)

   由于要计算损失函数：y_pred-y_true来更新权重，而y_pred的计算包含多个特征和多个权重，所以将特征值和权重引入矩阵，并补入偏置项方便weight0（逻辑回归模型中的常数项）的计算

  2.最开始没对数据进行标准化，因为LostFuction一直输出NAN，且一直报overflow，所以使用了Z-score标准化操作

  3.由于开始对数据进行了标准化，后续预测时需要进行逆标准化


* 可视化
![线性回归散点可视化](https://github.com/user-attachments/assets/f9556379-23bd-49c2-ac65-ccd3048b7aea)
  因为有三个特征，感觉可视化特征对标签的影响需要先固定一个特征，所以先给预测与真实的拟合情况

* 函数调用解释（基于chat）
  > 矩阵相关
  > > 1. np.hstack 用于水平拼接矩阵或者数组
  > > 2. np.ones() 用来创建一个全 ones 的数组或矩阵 简略语法: np.ones(shape)
  > >
  > >    eg: np.ones(3, 4)  创建一个 3x4 的全 ones 矩阵
  > > 4. X.shape[0] or X.shape[1] 用于获取矩阵X的行/列数，返回矩阵行数或者列数的属性
  > > 5. np.dot() 用于矩阵乘法
  > > 6. X.reshape(-1, 1) 如果X中有m个元素，将其转化为m×1的矩阵形式
  > > 7. X.T 用于计算一个矩阵 X 的转置
  > > 
  > 初始化相关
  > > 1.np.random.rand() 用来生成一组随机浮点数
  > 
  > >   eg:  # 生成一组随机浮点数（一维数组）   # 生成一个 3x4 的随机矩阵
  > 
  > >         print(np.random.rand(1))        print(np.random.rand(3, 4))
  > 
  > 其它
  > > 1.np.mean(X, axis=0) 补充axis=？，可计算矩阵 X 的某一列的平均值
  > 
  > > 2.X[X == 0] = 1 若X是个数组，索引X中为0的元素并更改为1
  >
  > > 3.Xmean[1]调用X中第二列元素的平均值
