import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
fp = r"D:\IQIYI Video\data.txt"
data = pd.read_csv(fp, sep=',', header=None)
data.columns = ['city', 'profit']
k0 = 0
k1 = 0
alpha = 0.02
m = len(data)

def yici(x):
    return k0 + k1 * x

def cost_function():
    cost = np.sum((yici(data['city']) - data['profit']) ** 2) / (2 * m)
    return cost

def gradient_descent():
    global k0, k1
    k0 = k0 - alpha * np.sum(yici(data['city']) - data['profit']) / m
    k1 = k1 - alpha * np.sum((yici(data['city']) - data['profit']) * data['city']) / m

plt.scatter(data['city'], data['profit'])
plt.xlabel('city')
plt.ylabel('profit')
plt.title('散点')
plt.show()

for i in range(1000):
    gradient_descent()
    if i % 100 == 0:
        print(f'Iteration {i}: Cost = {cost_function()}')

plt.scatter(data['city'], data['profit'])
plt.plot(data['city'], yici(data['city']), color='red')
plt.xlabel('city')
plt.ylabel('profit')
plt.title('优化后')
plt.show()
