	• 完成pycharm环境下的pytorch 在anaconda环境下的配置
        from __future__ import print_function
        import torch
        #经历了艰苦卓绝的奋斗，终于将环境配置完成了
	• 学习tensor
	1. 认识tensor的定义
x = torch.empty(5,3)
print(x)
# # 创建一个五行3列的随机列表
y = torch.rand(5,3)
print(y)
# # 创建一个五行三列的长型列表并且指定数据的类型为长型
# z = torch.zeros(5,3,dtype=torch.long )
# #long 类型前面只有一个数字，加上逗号
# print(z)
# x1 = torch.tensor([2.5,3.5])#直接通过输入数据创建张量
# print(x1)
# x = x.new_ones (5,3,dtype =torch.double)
# print(x)
#double 类型前面有一个逗号
#利用newmethod创建一个具有相同尺寸的新张量
# y1 = torch.rand_like(x,dtype= torch.float32)
# print(y1)
	2. 学习tensor的创建（创建随机矩阵，空矩阵。。。）
# #下面，要用x。size的 方式获取张量的形状
# #Torch。size的返回值本质上是一个元组
# a,b = x2.size()
# print("a = ", a) #a 向量表示的是矩阵的行数
# print("b = ", b) #b 向量表示的是矩阵的列数
	3. 学习tensor的运算
print(x + y)
#这是一种最简单的加法操作

#x下面展示第二种加法操作
print(torch.add(x,y))#注意的一点是，add的括号内的两个函数值，应当用，连接

# 下面展示第三种加法
#也就是说先设置一个空的张量，然后将函数的输出返回值赋予给那个量，最后输出被赋值的那一个量
result = torch.empty(5,3)
torch.add(x,y,out= result)
print(result)
	4. 学习tensor与numpy形式的转换
	5. 学习pytorch中的自动求导
	• 下周的学习计划
	1. 学习自动求导的原理
	2. 学习代码实操深层神经网络网络
	• 基于mnist数据集
	• 尝试改变激活函数和隐藏层的层数，观察对于准确性的影响
	3. 补习一些数学相关（矩阵方向的知识）
	• 矩阵的常用的运算
