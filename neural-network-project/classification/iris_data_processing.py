#Coding:UTF-8

import chainer
import numpy as np
import chainer.links as L
import chainer.functions as F

from chainer import Variable, Chain, optimizers
from sklearn import datasets

#Irisデータ読み込み
iris_data = datasets.load_iris()
#print(iris_data)

x = iris_data.data.astype(np.float32)
t = iris_data.target
n = t.size

#教師データの下処理
t_matrix = np.zeros(3 * n).reshape(n, 3).astype(np.float32)
for i in range(n):
    t_matrix[i, t[i]] = 1.0

print(t_matrix)

#訓練用データとテスト用データ　半分が訓練用データ、残りがテスト用データ
indexes = np.arange(n)
indexes_train = indexes[indexes % 2 != 0]
indexes_test = indexes[indexes % 2 == 0]

print(indexes)
print(indexes_train)
print(indexes_test)

x_train = x[indexes_train, :]#訓練用　入力
t_train = t_matrix[indexes_train, :]#訓練用　正解
x_test = x[indexes_test, :]#テスト用　入力
t_test = t_matrix[indexes_test]#テスト用　正解

print(x_train)
print(t_train)
print(x_test)
print(t_test)

x_train_v = Variable(x_train)
t_train_v = Variable(t_train)
x_test_v = Variable(x_test)