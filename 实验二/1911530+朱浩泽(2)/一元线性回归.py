import numpy as np 
import matplotlib.pyplot as plt 
import csv
import operator
with open('dataset_regression.csv') as csvfile:
     reader = csv.reader(csvfile)
     dataset = [row for row in reader]
dataset.pop(0)
for i in dataset:
    for m in range(3):
        i[m] = float(i[m])
print(dataset)
n = len(dataset)
sum_xy = 0
sum_x = 0
sum_y = 0
sum_xx = 0
for i in range(n):
    sum_xy += dataset[i][1] * dataset[i][2]
    sum_x += dataset[i][1]
    sum_y += dataset[i][2]
    sum_xx += dataset[i][1] * dataset[i][1]
a1 = (sum_xy - (sum_x * sum_y) / n) / (sum_xx - n * (sum_x / n  * sum_x / n))
a0 = sum_y / n - a1 * sum_x / n
a1 = round(a1, 4)
a0 = round(a0, 1)
print("回归方程为：y=", a1 , "x+" , a0)
xt = []
yt = []
for i in dataset:
    xt.append(i[1])
    yt.append(i[2])
loss = 0
for i in range(n):
    loss += (yt[i] - a0 - a1 * xt[i]) ** 2
loss = loss / (2 * n)
print("loss的值为：", loss)
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(xt, yt)
x = np.arange(-3, 4)
y = a1 * x + a0
plt.plot(x, y)
plt.show()
