import numpy as np 
import csv
import operator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
with open('winequality-white.csv') as csvfile:
     reader = csv.reader(csvfile)
     dataset = [row for row in reader]
dataset.pop(0)
y = []
for i in dataset:
    for m in range(len(i)):
        i[m] = float(i[m])
    y.append(i[-1])
    i.pop(-1)
    i.insert(0, 1)
x_train, x_test, y_train, y_test = train_test_split(dataset, y, test_size = 0.2) #划分训练集
#归一化
def feature_scaling(x):
    for i in range(len(x[0])):
        max = -float('inf')
        min = float('inf') 
        for m in range(len(x)):
            if x[m][i] > max:
                max = x[m][i]
            if x[m][i] < min:
                min = x[m][i]
        for m in range(len(x)):
            if max - min != 0:
                x[m][i] = (x[m][i] - min) / (max - min)
    return x
x_train = feature_scaling(x_train)
x_test = feature_scaling(x_test)
# theta = np.random.rand(len(x_train[0]))
# print(x_train[0])
# print(theta)
# print(theta * x_train)
# print(x_train)
# print(type(np.random.rand(len(x_train))))
def gradient_descent(x_train, y_train, x_test, y_test, learning_rate):
    loss = []
    theta = np.random.rand(len(x_train[0]))
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    for index in range(1000):
        gradients = x_train.T.dot(x_train.dot(theta) - y_train) / len(x_train)
        theta = theta - learning_rate * gradients
        MSE = ((np.dot(x_test, theta) - y_test) ** 2).sum() / len(x_test)
        loss.append(MSE)
    return theta, loss
ls = []
for i in range(1000):
    ls.append(i)
x = np.array(ls)
theta0, loss0 = gradient_descent(x_train, y_train, x_train, y_train, learning_rate = 0.5)
theta1, loss1 = gradient_descent(x_train, y_train, x_test, y_test, learning_rate = 0.5)
theta2, loss2 = gradient_descent(x_train, y_train, x_test, y_test, learning_rate = 0.3)
theta3, loss3 = gradient_descent(x_train, y_train, x_test, y_test, learning_rate = 0.1)
theta4, loss4 = gradient_descent(x_train, y_train, x_test, y_test, learning_rate = 0.01)
theta5, loss5 = gradient_descent(x_train, y_train, x_test, y_test, learning_rate = 0.001)
# 画散点图
colors0 = '#000000'
colors1 = '#00CED1' #点的颜色
colors2 = '#DC143C'
colors3 = '#66CDAA'
colors4 = '#BEBEBE'
colors5 = '#00FA9A'
area = np.pi * 0.5**2  # 点面积
plt.scatter(x, loss0, s=area, c=colors0, alpha=0.4, label='train')
plt.scatter(x, loss1, s=area, c=colors1, alpha=0.4, label='learning_rate = 0.5')
plt.scatter(x, loss2, s=area, c=colors2, alpha=0.4, label='learning_rate = 0.3')
# plt.scatter(x, loss3, s=area, c=colors3, alpha=0.4, label='learning_rate = 0.1')
# plt.scatter(x, loss4, s=area, c=colors4, alpha=0.4, label='learning_rate = 0.01')
# plt.scatter(x, loss5, s=area, c=colors5, alpha=0.4, label='learning_rate = 0.001')
plt.legend()
plt.show()