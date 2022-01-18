import numpy as np 
import matplotlib.pyplot as plt
import math 
def f(m, cnt):
    x = [] 
    y = []
    for i in range(cnt):
        x.append(m[i][0])
        y.append(m[i][1])
    return x, y

def random_x(cnt1, cnt2, cnt3, name): 
    cov = [[1, 0], [0, 1]]
    a1 = np.random.multivariate_normal((1, 1), cov, cnt1)
    a2 = np.random.multivariate_normal((4, 4), cov, cnt2)
    a3 = np.random.multivariate_normal((8, 1), cov, cnt3)
    colors0 = '#000000'
    colors1 = '#00CED1' #点的颜色
    colors2 = '#DC143C'
    area = np.pi   # 点面积
    x, y = f(a1, cnt1)
    plt.scatter(x, y, s=area, c=colors0, alpha=0.4)
    x, y = f(a2, cnt2)
    plt.scatter(x, y, s=area, c=colors1, alpha=0.4)
    x, y = f(a3, cnt3)
    plt.scatter(x, y, s=area, c=colors2, alpha=0.4)
    ls = []
    result = []
    for i in range(cnt1):
        ls.append(a1[i])
        result.append(1)
    for i in range(cnt2):
        ls.append(a2[i])
        result.append(2)
    for i in range(cnt3):
        ls.append(a3[i])
        result.append(3)
    plt.figure(num = name) 
    return ls, result

x1, c1 = random_x(333, 333, 334, "X_1数据集散点图")
x2, c2 = random_x(600, 300, 100, "X_2数据集散点图")


def Normal_distribution(data, m):
    m = np.array(m)
    cov = np.array([[1, 0], [0, 1]])
    return 1 / math.sqrt((2 * np.pi) ** 2 * 1) * math.exp(-0.5 * np.dot((data - m).T, (data - m))) 

def Classification(data, mean1, mean2, mean3):
    result = []
    for d in data:
        t1 = Normal_distribution(d, mean1)
        t2 = Normal_distribution(d, mean2)
        t3 = Normal_distribution(d, mean3)
        i = 1
        max = t1
        if t2 > max:
            max = t2
            i = 2
        if t3 > max:
            max = t3
            i = 3
        result.append(i)
    return result

def simrate(ls1, ls2):
    num = 0
    l = len(ls1)
    for i in range(l):
        if ls1[i] != ls2[i]:
            num += 1
    return format(num / l, '.2%') 

def show_result(data, result, name):
    colors0 = '#000000'
    colors1 = '#00CED1' #点的颜色
    colors2 = '#DC143C'
    area = np.pi   # 点面积
    x1 = []
    x2 = []
    x3 = []
    y1 = []
    y2 = []
    y3 = []
    for i in range(len(data)):
        if result[i] == 1:
            x1.append(data[i][0])
            y1.append(data[i][1])
        elif result[i] == 2:
            x2.append(data[i][0])
            y2.append(data[i][1])
        elif result[i] == 3:
            x3.append(data[i][0])
            y3.append(data[i][1])
    plt.scatter(x1, y1, s=area, c=colors0, alpha=0.4)
    plt.scatter(x2, y2, s=area, c=colors1, alpha=0.4)
    plt.scatter(x3, y3, s=area, c=colors2, alpha=0.4)
    plt.figure(num = name)

result1 = Classification(x1, (1, 1), (4, 4), (8, 1))
result2 = Classification(x2, (1, 1), (4, 4), (8, 1))

Error_rate1 = simrate(result1, c1)
Error_rate2 = simrate(result2, c2)

show_result(x1, result1, "X_1最大后验概率规则分类图")
show_result(x2, result2, "X_2最大后验概率规则分类图")


# plt.show()

#print("\nX_1数据集使用最大后验概率规则进行分类的错误率是", Error_rate1)
#print("X_2数据集使用最大后验概率规则进行分类的错误率是", Error_rate2)




def guss(x, data, h):
    n = len(data)
    result = 0
    for i in range(n):
        result += math.exp(- (math.sqrt((x[0] - data[i][0]) ** 2 + (x[1] - data[i][1]) ** 2)) / (2 * h * h))
    result = result / (n * math.sqrt(2 * np.pi * h * h))
    return result

def Classification_2(data, m1, m2, m3, h, r):
    result = []
    data1 = data[0 : m1]
    data2 = data[m1 : m1+m2]
    data3 = data[m1+m2 : m1+m2+m3]
    for d in data:
        t1 = guss(d, data1, h)
        t2 = guss(d, data2, h)
        t3 = guss(d, data3, h)
        i = 1
        max = t1
        if t2 > max:
            max = t2
            i = 2
        if t3 > max:
            max = t3
            i = 3
        result.append(i)
    return simrate(result, r)

r1_1 = Classification_2(x1, 334, 333, 333, 0.1, c1)
r1_5 = Classification_2(x1, 334, 333, 333, 0.5, c1)
r1_10 = Classification_2(x1, 334, 333, 333, 1, c1)
r1_15 = Classification_2(x1, 334, 333, 333, 1.5, c1)
r1_20 = Classification_2(x1, 334, 333, 333, 2, c1)

r2_1 = Classification_2(x2, 600, 300, 100, 0.1, c2)
r2_5 = Classification_2(x2, 600, 300, 100, 0.5, c2)
r2_10 = Classification_2(x2, 600, 300, 100, 1, c2)
r2_15 = Classification_2(x2, 600, 300, 100, 1.5, c2)
r2_20 = Classification_2(x2, 600, 300, 100, 2, c2)
print("X_1数据集利用似然率测试规则在h = 0.1情况下分类错误率是 ", r1_1)
print("X_1数据集利用似然率测试规则在h = 0.5情况下分类错误率是 ", r1_5)
print("X_1数据集利用似然率测试规则在h = 1.0情况下分类错误率是 ", r1_10)
print("X_1数据集利用似然率测试规则在h = 1.5情况下分类错误率是 ", r1_15)
print("X_1数据集利用似然率测试规则在h = 2.0情况下分类错误率是 ", r1_20)

print("X_2数据集利用似然率测试规则在h = 0.1情况下分类错误率是 ", r2_1)
print("X_2数据集利用似然率测试规则在h = 0.5情况下分类错误率是 ", r2_5)
print("X_2数据集利用似然率测试规则在h = 1.0情况下分类错误率是 ", r2_10)
print("X_2数据集利用似然率测试规则在h = 1.5情况下分类错误率是 ", r2_15)
print("X_2数据集利用似然率测试规则在h = 2.0情况下分类错误率是 ", r2_20)