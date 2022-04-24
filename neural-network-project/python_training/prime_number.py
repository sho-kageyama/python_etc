#Coding:UTF-8

import math
import numpy as np
import matplotlib.pyplot as plt
import chainer
import chainer.functions as F

def prime(num):
    if num == 2:
        return True
    elif num <= 2 or num % 2 == 0:
        return False
    sqrtnum = math.sqrt(num)
    count = 3
    while count <= sqrtnum:
          if num % count == 0:
              return False
          else:
              count += 2

    return True


a = []

for i in range(1, 10000):
    if prime(i):
       a.append(i)
       print(i)
print(len(a))


