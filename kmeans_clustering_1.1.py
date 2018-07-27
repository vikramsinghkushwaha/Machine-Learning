import numpy as np
from sklearn import preprocessing,cross_validation, neighbors
import pandas as pd

df = pd.read_csv('/home/vicky/Documents/breast-cancer-wisconsin.data.txt')
df.replace('?',-99999,inplace = True)
df.drop(['id'],1,inplace = True)

x = np.array(df.drop(['class'],1))
y = np.array(df['class'])

x_train,x_test,y_train,y_test = cross_validation.train_test_split(x,y,test_size = 0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(x_train,y_train)

accuracy = clf.score(x_test,y_test)
print accuracy

testing = np.array([[4,2,1,2,1,4,1,2,1]])
testing = testing.reshape(len(testing),-1)
prediction = clf.predict(testing)

print prediction
