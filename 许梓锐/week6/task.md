# TASK
## 不会上传文件，所以就直接贴过来了

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from docx import Document
import pandas as pd


doc = Document('D:\俱乐部专用\data.docx')


data_x = []
data_y=[]

for para in doc.paragraphs:

    row = para.text.strip().split(',')  
    if len(row)==2:
      
            value=float(row[0].strip())
            value1=float(row[1].strip())
            data_x.append(value)
            data_y.append(value1)
keys = ["area", "price"]  
values = [data_x, data_y] 

my_dict = dict(zip(keys, values))

print(my_dict)
df=pd.DataFrame(my_dict)
x=df[["area"]]
y=df["price"]
import pandas as pd
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)  # 计算均方误差（MSE）
r2 = r2_score(y_test, y_pred)  # 计算 R² 得分（决定系数）
import matplotlib.pyplot as plt

plt.scatter(x, y, color='blue', label='数据点')
plt.plot(x, model.predict(x), color='red', label='回归线')
plt.xlabel('人口数量')
plt.ylabel('利润')
plt.legend()
plt.show()
