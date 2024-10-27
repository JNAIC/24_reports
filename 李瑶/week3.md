## week3周报
# 还在学习python的第一阶段中，现在已初步了解模块和异常等方面的知识，类也是大致懂了，顺便把上周不懂的__main__给弄懂了，大致是指只有在当前python文件（也就是模块当中）才能使name=main，从而时只有在当前文件中可以执行main下面的代码
# def print_file_into(name):
#    try:
#        f = open("name","r",encoding="UTF_8")
#   except:
#        print("不存在此文件")
#        f = open("name","w",encoding="UTF_8")
# def append_to_file(name,data):
#    f = open("name","a",encoding="UTF_8")
#    f.write(data)
# 以上是一个简单的捕获异常的代码
# 初步了解神经网络的激活函数，简单来说就是将输入值通过一些法则转换为我们需要的适合的输出值，类似函数的按照法则让原象变成象。
# 创造两种简单的激活函数的图像
# import numpy as np
# import matplotlib.pyplot as plt
# def step_fuction(x):
#    return np.array(x>0,dtype=np.int64)
# def sigmoid_fuction(x):
#    return 1/(1+np.exp(-x)) 
# x1 = np.arange(-5.0,5.0,0.1)
# x2 = np.arange(-5.0,5.0,0.1) 
# y1 = step_fuction(x1) 
# y2 = sigmoid_fuction(x2)
# plt.plot(x1,y1,linestyle='--',label='step')
# plt.plot(x2,y2,label='sigmoid')
# plt.ylim(-0.1,1.1)
# plt.show()
