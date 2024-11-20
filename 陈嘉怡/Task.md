* 代码
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.read_csv('data.txt', header=None) 
x = data.iloc[:, 0]
y = data.iloc[:, 1]
X = np.c_[np.ones((len(x),1)),x]
beta=np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y.values)

plt.scatter(x,y)
plt.plot(x,beta[0]+beta[1]*x,c='black')
plt.show()
```
* 可视化
  ![image](https://github.com/user-attachments/assets/e672cc46-c334-4306-8d56-28771ea31146)
