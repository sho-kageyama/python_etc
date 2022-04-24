#coding: UTF-8
import numpy as np

a = np.arange(6).reshape(2,3)
print(a)

b = np.arange(6).reshape(2,3)
print(b)

c = np.hstack([a,b])
print(c)

d = np.vstack([a,b])
print(d)

e = np.arange(20).reshape(4,5)
print(e)

f = e[[0, 2], :]
print(f)

g = e[:,[0,2,4]]
print(g)

e[e % 2 == 0] = 0
e[e % 2 != 0] = 1
print(e)

