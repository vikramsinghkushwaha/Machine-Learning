import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, cross_validation,svm
from sklearn.linear_model import LinearRegression

df = quandl.get('wiki/googl')

print('Length of df is: ', len(df))
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = ((df['Adj. High'] - df['Adj. Close']) / df['Adj. Close']) * 100.0
df['Daily_change'] = ((df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open']) * 100.0
df = df[['Adj. Close', 'Adj. Volume', 'HL_PCT', 'Daily_change']]
forecast_col = 'Adj. Close'
df.fillna(-99999, inplace = True)

forecast_out = int(math.ceil(0.01 * len(df)))
print("The number of days forecasted is: ", forecast_out)
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace = True)
#print(df)

X = np.array(df.drop(['label'],1))
y = np.array(df['label'])

X = preprocessing.scale(X)

#X = X[:-forecast_out+1]
#df.dropna(inplace = True)
print (len(X),len(y))
X_train, X_test,y_train, y_test = cross_validation.train_test_split(X,y,test_size = 0.2)

clf = LinearRegression()
clf.fit(X_train,y_train)
accuracy = clf.score(X_test, y_test)

print("The accuracy is",accuracy)
