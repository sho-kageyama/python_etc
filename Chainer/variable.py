#coding: UTF-8
import chainer
import numpy as np
from chainer import Variable

input_array = np.array([[1,2,3],[3,4,5]], dtype=np.float32)
print(input_array)

x = Variable(input_array)
print(x.data)

#variable計算
#y = x * 2 + 1
y = x ** 2 + 2 * x  + 1 #微分計はy = 2 * x + 2
print(y.data)

#微分値を求める
#要素が複数の場合はy.gradに初期値が必要
y.grad = np.ones((2,3), dtype=np.float32)
#x→yに遡って微分値を求める
y.backward()
#y = x * 2 + 1 の微分計は y = 2
print(x.grad)

z = x ** 2 + 6 * x + 9
print(z.data)
z.grad = np.ones((2,3), dtype=np.float32)
z.backward()
print(x.grad)