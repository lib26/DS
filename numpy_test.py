import random

import numpy as np
from numpy import pi, ones, zeros, identity
from numpy import newaxis

import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import math

a = np.arange(15).reshape(3, 5)
a = np.array([[0, 1, 2, 3, 4],
              [5, 6, 7, 8, 9],
              [10.1, 11, 12, 13, 14]], dtype=np.float64)
print('a:', a)

b = np.ones((3, 4), dtype=np.int16)
print('b:', b)

c = np.arange(1, 10.1, 3)
print('c:', c)

d = np.linspace(0, 2 * pi, 3)
print('d:', d)

sin = np.sin(d)
print('sin:', sin)

e = np.floor(10 * np.random.random((2, 12)))
print('e:', e)
print('hsplit:', np.hsplit(e, 3))

langs = ['C', 'C++', 'Java', 'Python', 'PHP']
students = [23, 17, 35, 29, 12]
plt.bar(langs, students)
plt.show()

a = np.array([[2, 4, 6, 8],
              [1, 3, 5, 7]])
b = np.array([[1, 2],
              [2, 3],
              [3, 4],
              [4, 5]])
print('dot:',np.dot(a, b))

