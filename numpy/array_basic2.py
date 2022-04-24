#coding: UTF-8
import numpy as np

a = np.arange(42).reshape(6,7)
print(a)
print(np.shape(a))
print(np.size(a))

b = np.random.rand(10)
print(b)

c = np.zeros(11)
print(c)
d = np.ones(5)
print(d)

e = np.random.permutation(np.arange(12))
print(e)

(r, col) = a.shape
print(r)
print(col)