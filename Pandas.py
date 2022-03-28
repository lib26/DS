import pandas as pd
import numpy as np

# create Panda (4,4)  (3. ? 2. 5. * 4. 5. 6. + 3. 2. & 5. ? 7. !)
df = pd.DataFrame({'a':[3.,'?', 2., 5.], 'b':['*', 4., 5., 6.], 'c':['+', 3., 2., '&'], 'd': [5., '?', 7, '!']})
print()

# Display the DataFrame
print(df)
print()

# Replace any non-numeric value with NaN
df = df.replace({'?':np.nan, '*':np.nan, '+':np.nan, '&':np.nan, '!':np.nan})
print()

# Display the DataFrame
print(df)
print()

# Apply the following functions one at a time in sequence to the DataFrame, and display the DataFrame after applying each function.
# isna with any, and sum
print(df.isna().any())
print(df.isna().sum())
print()

# dropna with howany, howall, thresh1, thresh2
print("---------dropna/axis=0---------")
print(df.dropna(axis=0, how='all'))
print(df.dropna(axis=0, how='any'))
print(df.dropna(axis=0, how='any', thresh=1))
print(df.dropna(axis=0, how='any', thresh=2))
print("---------dropna/axis=1---------")
print(df.dropna(axis=1, how='all'))
print(df.dropna(axis=1, how='any'))
print(df.dropna(axis=1, how='any', thresh=1))
print(df.dropna(axis=1, how='any', thresh=2))
print()

# fillna with 100, mean, median
print("---------fillna---------")
print(df.fillna(100))
print(df.fillna(df.mean()))
print(df.fillna(df.median()))
print()
print()

# ffill, bfill
print("---------fillna(ffill/bfill)---------")
print(df.fillna(axis=0, method='ffill'))
print(df.fillna(axis=0, method='bfill'))
print(df.fillna(axis=1, method='ffill'))
print(df.fillna(axis=1, method='bfill'))
print()
