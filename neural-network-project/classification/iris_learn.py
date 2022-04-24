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


#訓練用データとテスト用データ　半分が訓練用データ、残りがテスト用データ
indexes = np.arange(n)
indexes_train = indexes[indexes % 2 != 0]
indexes_test = indexes[indexes % 2 == 0]



x_train = x[indexes_train, :]#訓練用　入力
t_train = t_matrix[indexes_train, :]#訓練用　正解
x_test = x[indexes_test, :]#テスト用　入力
t_test = t_matrix[indexes_test]#テスト用　正解



x_train_v = Variable(x_train)
t_train_v = Variable(t_train)
x_test_v = Variable(x_test)

#Chainの記述
class IrisChain(Chain):
    def __init__(self):
        super(IrisChain, self).__init__(
            l1 = L.Linear(4,6),
            l2 = L.Linear(6,6),
            l3 = L.Linear(6,3)
        )

    def predict(self, x):
        h1 = F.sigmoid(self.l1(x))
        h2 = F.sigmoid(self.l2(h1))
        h3 = self.l3(h2)
        return h3

#モデルとoptimizerの記述
model = IrisChain()
optimizer = optimizers.Adam()
optimizer.setup(model)

#学習
for i in range(10000):
    #モデルの勾配を消去し、y_train_vにx_train_vの出力値
    model.cleargrads()
    y_train_v = model.predict(x_train_v)

    #損失関数による誤差の計算、今回は平均二乗誤差
    loss = F.mean_squared_error(y_train_v, t_train_v)
    loss.backward()#backward関数により逆電波

    #optimizerによる重みの更新
    optimizer.update()
