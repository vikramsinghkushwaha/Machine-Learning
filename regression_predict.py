import pandas as pd
import quandl, math
import datetime, math, pickle
import numpy as np
from sklearn import preprocessing, cross_validation,svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

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

X_lately = X[-forecast_out:]
#df.dropna(inplace = True)
#print (len(X),len(y))
X_train, X_test,y_train, y_test = cross_validation.train_test_split(X,y,test_size = 0.2)

clf = LinearRegression()
clf.fit(X_train,y_train)
accuracy = clf.score(X_test, y_test )

with open('linearregression_1.1.pickle','wb') as f:   #This saves the trained classifier in pickle file
    pickle.dump(clf,f)                                #Comment the above and this line after running once, it will be saved    

pickle_in = open('linearregression_1.1.pickle','rb')

clf = pickle.load(pickle_in)
print("The accuracy is",accuracy)

predicted_value = clf.predict(X_lately)
print(predicted_value, accuracy, forecast_out)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in predicted_value:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc = 4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
