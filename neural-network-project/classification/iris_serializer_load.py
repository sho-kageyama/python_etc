#Coding:UTF-8

import chainer
import numpy as np
import chainer.links as L
import chainer.functions as F

from chainer import Variable, Chain, optimizers, serializers
from chainer.training import extensions
from chainer.datasets import tuple_dataset
from chainer import training, iterators


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


#訓練用データとテスト用データ　半分が訓練用データ、残りがテスト用データ
indexes = np.arange(n)
indexes_train = indexes[indexes % 2 != 0]
indexes_test = indexes[indexes % 2 == 0]



x_train = x[indexes_train, :]#訓練用　入力
t_train = t_matrix[indexes_train, :]#訓練用　正解
x_test = x[indexes_test, :]#テスト用　入力
t_test = t_matrix[indexes_test]#テスト用　正解



train = tuple_dataset.TupleDataset(x_train, t_train)

x_test_v = Variable(x_test)


#Chainの記述
class IrisChain(Chain):
    def __init__(self):
        super(IrisChain, self).__init__(
            l1 = L.Linear(4,6),
            l2 = L.Linear(6,6),
            l3 = L.Linear(6,3)
        )


    def __call__(self, x, t):
        return F.mean_squared_error(self.predict(x), t)

    def predict(self, x):
        h1 = F.sigmoid(self.l1(x))
        h2 = F.sigmoid(self.l2(h1))
        h3 = self.l3(h2)
        return h3

#モデルとoptimizerの記述
model = IrisChain()
optimizer = optimizers.Adam()
optimizer.setup(model)

#モデルの読み込み
serializers.load_npz("my_iris.npz", model)

#テスト
model.cleargrads()
y_test_v = model.predict(x_test_v)
y_test = y_test_v.data
print(t_test[0])

#正解数のカウント
correct = 0
rowCount = y_test.shape[0]
for i in range(rowCount):
    maxIndex = np.argmax(y_test[i, :]) #ng.argmax関数は最大の要素のインデックスを表す
    print(y_test[i, :], maxIndex)
    if maxIndex == np.argmax(t_test[i]):
        correct += 1


#正解率
print("Correct:", correct, "Total:", rowCount, "Acuuracy:", correct / rowCount * 100, "%")


