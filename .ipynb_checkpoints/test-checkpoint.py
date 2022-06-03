#!/usr/bin/env python
import math
import matplotlib.pyplot as plt
plt.switch_backend('agg')

p1 = [1,2,3,4]
p2 = [3,4,5,6]

plt.scatter(p1,p2)
plt.savefig('Graphs/test.png')