#This codes contain calculation of r squared mean. This is used for testing or to get an idea of how best fit our line is
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean
from matplotlib import style
import random

style.use('fivethirtyeight')

#xs = np.array([1,2,3,4,5,6], dtype = np.float64)
#ys = np.array([5,4,6,5,6,7], dtype = np.float64)

#Creation of random dataset 
def create_dataset(hm,variance,step=2,correlation=False):
	val = 1
	ys = []
	for i in range(hm):
		y = val + random.randrange(-variance,variance)
		ys.append(y)
	if correlation and correlation == 'pos':
		val += step
	elif correlation and correlation == 'neg':
		val -= step
	xs = [i for i in range(len(ys))]
	return np.array(xs,dtype = np.float64),np.array(ys,dtype = np.float64)

#Calculation of squared mean theorem
def squared_error(ys_orig,ys_line):
	return sum((ys_line - ys_orig)**2)

#The main formula for calculation of the coreectness of our best fit line
def coefficient_determination(ys_orig,ys_line):
	y_mean_line = [mean(ys_orig) for y in ys_orig]
	squared_error_regr = squared_error(ys_orig,ys_line)
	squared_error_y_mean = squared_error(ys_orig,y_mean_line)
	return (1- (squared_error_regr/squared_error_y_mean) )
 #The best fit line
def slope_intercept(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) / ((mean(xs)*mean(xs)) -mean(xs*xs) ))
    b = mean(ys) - m*mean(xs)
    return m,b

xs,ys = create_dataset(10,10,2,correlation='pos')

m,b = slope_intercept(xs,ys)
print(m,b)

regression_line = [(m*x) + b for x in xs]

r_squared = coefficient_determination(ys,regression_line) 
print "squared error is : ",r_squared

#Prediction of value
predict_x = 8
predict_y = m*predict_x + b

#Systematic display of results
plt.scatter(predict_x, predict_y, color = 'g')
plt.scatter(xs,ys, color = 'b')
plt.plot(xs, regression_line)
plt.show()
