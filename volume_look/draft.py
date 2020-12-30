import pandas as pd
import numpy as np
import time

df = pd.DataFrame()
df['a'] = [1, 2, 3, 4, 5]
df['b'] = [-2, 3, 6, -2, -6]
df['c'] = [np.nan, np.nan, np.nan, 2, np.nan]

p = df['c'].first_valid_index()
df['c'].loc[p:] = df['c'].loc[p:].fillna(0)
print(df)