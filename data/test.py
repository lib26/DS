import numpy as np
import pandas as pd
import pickle
import os
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings('ignore')


pd.set_option('display.max_columns', 16) # to show all column

# data_type
df = pd.read_csv("vgsales.csv")

idx = df[df['User_Score']=='tbd'].index
df = df.drop(idx)
df = df.dropna(axis=0)

print(df['User_Score'].dtype)
df['User_Score'] = df['User_Score'].astype(float, errors = 'raise')
print(df['User_Score'].dtype)










