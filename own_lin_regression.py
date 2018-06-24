import matplotlib.pyplot as plt
import numpy as np
from statistics import mean
from matplotlib import style

style.use('fivethirtyeight')

xs = np.array([1,2,3,4,5,6], dtype = np.float64)
ys = np.array([5,4,6,5,6,7], dtype = np.float64)

def slope_intercept(xs,ys):
    m = (mean(xs)*mean(ys) - mean(xs*ys))/(mean(xs))/(mean(xs)*mean(xs) -mean(xs*xs) )
    b = mean(ys) - m*mean(xs)
    return m,b

m,b = slope_intercept(xs,ys)
print(m,b)

regression_line = [(m*x) + b for x in xs]

plt.scatter(xs,ys)
plt.plot(xs, regression_line)
plt.show()
