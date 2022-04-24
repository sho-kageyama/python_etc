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

print(x)
print(t)
print(n)