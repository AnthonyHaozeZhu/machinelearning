import numpy as np 
import csv
import operator
train = []
train_result = []
test = []
test_right = []
# 导入数据
with open('semeion_train.csv') as csvfile:
    reader = csv.reader(csvfile)
    rows= [row for row in reader]
for i in rows:
    ls = [] 
    temp = i[0].split()
    num = 0
    for m in temp[:-10]:
        m = float(m)
        ls.append(m)
    for m in temp[-10:]:
        m = int(m)
        if m == 1:
            train_result.append(num)
        num += 1
    train.append(ls)
with open('semeion_test.csv') as csvfile:
    reader = csv.reader(csvfile)
    rows= [row for row in reader]
for i in rows:
    ls = [] 
    temp = i[0].split()
    for m in temp[:-10]:
        m = float(m)
        ls.append(m)
    num = 0
    for m in temp[-10:]:
        m = int(m)
        if m == 1:
            test_right.append(num)
        num += 1
    test.append(ls)
# print(test_right)
#实现KNN算法
def KNN(inX, train, train_result, k):
    size = len(train)
    train = np.asarray(train)
    inX = np.asarray(inX)
    result = []
    for X in inX:
        exp = np.tile(X, (size , 1))
        differ = exp - train
        square = differ ** 2
        distance = (square.sum(axis = 1)) ** 0.5
        # print(distance)
        sorted_index = distance.argsort()
        temp = [0] * 10 
        for m in sorted_index[:k]:
            temp[train_result[m]] += 1
            temp = np.asarray(temp)
        result.append(temp.argsort()[-1])
    return result
#利用自己编写的KNN算法对手写数字进行分类
result1 = KNN(test, train, train_result, 1)
result3 = KNN(test, train, train_result, 3)
result5 = KNN(test, train, train_result, 5)
#求相似度
def simrate(ls1, ls2):
    num = 0
    l = len(ls1)
    for i in range(l):
        if ls1[i] == ls2[i]:
            num += 1
    return format(num / l, '.2%')
print("k = 1时的准确率是：", simrate(result1, test_right))    
print("k = 3时的准确率是：", simrate(result3, test_right))
print("k = 5时的准确率是：", simrate(result5, test_right))
#与sklearn库中做对比
from sklearn.neighbors import KNeighborsClassifier
knn1 = KNeighborsClassifier(1)
knn1.fit(train, train_result)
knn3 = KNeighborsClassifier(3)
knn3.fit(train, train_result)
knn5 = KNeighborsClassifier(5)
knn5.fit(train, train_result)
resultsk1 = knn1.predict(test)
resultsk3 = knn3.predict(test)
resultsk5 = knn5.predict(test)
print("sklearn中k = 1时的准确率是：", simrate(resultsk1, test_right))
print("sklearn中k = 3时的准确率是：", simrate(resultsk3, test_right))
print("sklearn中k = 5时的准确率是：", simrate(resultsk5, test_right))