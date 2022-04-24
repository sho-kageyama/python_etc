#coding:UTF-8
import chainer
import numpy as np
import chainer.links as L
from chainer import Variable


#Links liner Link (入力に係数をかけたものと、バイアスを足し合わせる) 関数によりオブジェクト1を作成
l = L.Linear(3,2)
print(l.W.data)
print(l.b.data)

#オブジェクトによりｙを計算
input_array = np.array([[1,2,3],[4,5,6]], dtype=np.float32)
x = Variable(input_array)
y = l(x)
print(y.data)

#lの勾配をゼロに初期化
l.cleargrads()

#y→lに遡って微分を計算
#y.grad = np.ones((1,2), dtype=np.float32)
y.grad = np.ones((2,2), dtype=np.float32)
y.backward()
print(l.W.grad)
print(l.b.grad)


a = L.Linear(4,4)
print(a.W.data)
print(a.b.data)

input = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]],dtype=np.float32)
beta = Variable(input)
p = a(beta)
print(p.data)

a.cleargrads()

p.grad = np.ones((4,4), dtype=np.float32)
p.backward()
print(a.W.grad)
print(a.b.grad)