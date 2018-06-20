import pandas as pd
import quandl

df = quandl.get('wiki/googl')
pd.set_option('display.max_columns',None)
print(df.head())
