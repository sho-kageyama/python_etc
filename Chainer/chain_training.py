#coding: UTF-8
import chainer
import numpy as np
from chainer import Variable, Chain
import chainer.links as L

#関数で記述
l1 = L.Linear(4,3)
l2 = L.Linear(3,2)

def my_forward(x):
    h = l1(x)
    return l2(h)

#動作検証
input_array = np.array([[1,2,3,4]], dtype=np.float32)
x = Variable(input_array)
y = my_forward(x)
print(y.data)

#クラスで記述
class MyClass():
    def __init__(self):
        self.l1 = L.Linear(4,3)
        self.l2 = L.Linear(3,2)

    def forward(self, x):
        h = self.l1(x)
        return self.l2(h)

#動作検証
input_array = np.array([[1,2,3,4]], dtype=np.float32)
x = Variable(input_array)
my_class = MyClass()
y = my_class.forward(x)
print(y.data)

#Chainクラスを継承
class MyChain(Chain):
    def __init__(self):
        super().__init__(
            l1 = L.Linear(4,3),
            l2 = L.Linear(3,2),
        )
    def __call__(self, x):
        h = self.l1(x)
        return self.l2(h)

#動作検証
input_array = np.array([[890,567,555,823]], dtype=np.float32)
x = Variable(input_array)
my_chain = MyChain()
y = my_chain(x)
print(y.data)
