import numpy as np 
import math
import random
import csv
import operator


with open('wine.data') as csvfile:
     reader = csv.reader(csvfile)
     dataset = [row for row in reader]


#按照7:3的比例层抽样划分测试集和训练集(125 53)
#42 50 33 
x_train = []
y_train = []
x_test = []
y_test = []
seed = []
train = random.sample(dataset[0: 59], 42)
train = train + random.sample(dataset[59: 59 + 71], 50)
train = train + random.sample(dataset[59 + 71: -1], 33)
test = [i for i in dataset if i not in train]

def to_float(dataset):
    y = []
    for i in dataset:
        for m in range(len(i)):
            i[m] = float(i[m])
        y.append(int(i[0]))
        i.pop(0)
    return dataset, y

x_train, y_train = to_float(train)
x_test, y_test = to_float(test)


def Bayes(data, p, avg, var):
    result = p
    for i in range(len(data)):
        result *=  1 / (math.sqrt(2 * math.pi * var[i])) * math.exp(-((data[i] - avg[i])**2) / (2 * var[i]))
    return result

def classifier(x_train, x_test):
    result = []
    x_train = np.array(x_train)
    avg1 = x_train[:42].mean(axis = 0)
    var1 = x_train[:42].var(axis = 0)
    avg2 = x_train[42 : 42 + 50].mean(axis = 0)
    var2 = x_train[42 : 42 + 50].var(axis = 0)
    avg3 = x_train[42 + 50 : ].mean(axis = 0)
    var3 = x_train[42 + 50 : ].var(axis = 0)
    for i in range(len(x_test)):
        temp = 1
        max = Bayes(x_test[i], 59 / (59 + 71 + 48), avg1, var1)
        if Bayes(x_test[i], 71 / (59 + 71 + 48), avg2, var2) > max:
            temp = 2
            max = Bayes(x_test[i], 71 / (59 + 71 + 48), avg2, var2)
        if Bayes(x_test[i], 48 / (59 + 71 + 48), avg3, var3) > max:
            temp = 3
        result.append(temp)
    return result

def simrate(ls1, ls2):
    num = 0
    l = len(ls1)
    for i in range(l):
        if ls1[i] == ls2[i]:
            num += 1
    return format(num / l, '.2%')

predict = classifier(x_train, x_test)

print("分类的准确率是", simrate(predict, y_test))

def confuse_maxtria(predict, fact):
    ls = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for i in range(len(predict)):
        ls[fact[i] - 1][predict[i] - 1] += 1
    return ls

print("混淆矩阵是:", confuse_maxtria(predict, y_test))

def get_feature(confuse_maxtria):
    for index in range(len(confuse_maxtria)):
        truth = confuse_maxtria[index][index]
        total = 0
        total2 = 0
        for i in range(len(confuse_maxtria)):
            total += confuse_maxtria[index][i]
        for i in range(len(confuse_maxtria)):
            total2 += confuse_maxtria[i][index]
        precision = truth / total
        recall = truth / total2
        f_rate = 2 * precision * recall / (precision + recall)
        print("类别", index + 1, "的精度为", precision, "，召回率为", recall, "，F值为", f_rate)

get_feature(confuse_maxtria(predict, y_test))