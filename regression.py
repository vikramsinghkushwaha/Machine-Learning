import pandas as pd
import quandl

df = quandl.get('wiki/googl')
pd.set_option('display.max_columns',None)
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = ((df['Adj. High'] - df['Adj. Close']) / df['Adj. Close']) * 100.0
df['Daily_change'] = ((df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open']) * 100.0
pd.set_option('display.max_columns', None)
df = df[['HL_PCT', 'Daily_change', 'Adj. Close', 'Adj. Volume']]
print(df.head())
