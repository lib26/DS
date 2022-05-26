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

# Data exploration
print("-------------data type -------------")
print(df.dtypes)
print()
print("-------------shape--------------")
print(df.shape)
print()
print("-------------all column ------------")
print(df.describe())
print()
print("-------------categorical data---------------")
categorical = df.dtypes[df.dtypes == "object"].index
print(categorical)
print()
print("-------------categorical data ------------")
print(df[categorical].describe())
print()


# Data cleaning
# Show all nan values
# As you can see Score, Count, Developer and Rating has the most NAN values
print("-----Number of NAN -------")
print(df.isnull().sum())
print()


# First, discard all 'features' that are not related to Rating and sales volume,
# and then discard rows with NA values.
df2 = df.drop(['Name', 'Platform', 'Year_of_Release', 'Developer',
               'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales',
               'Critic_Count', 'User_Count'], axis=1)
df2 = df2.dropna(axis=0)
idx = df2[df2['User_Score']=='tbd'].index
df2 = df2.drop(idx)
df2['User_Score'] = df2['User_Score'].astype(float, errors = 'raise')


# Global sales, critic score scatter -> 평점과 판매량 사이에 연관이 크게 없는듯..
plt.scatter(df2['Critic_Score'], df2['Global_Sales'])
plt.xlabel("Critic_Score")
plt.ylabel("Global_Sales")
plt.show()

# Global sales, user score scatter -> 평점과 판매량 사이에 연관이 크게 없는듯..
plt.scatter(df2['User_Score'], df2['Global_Sales'])
plt.xlabel("User_Score")
plt.ylabel("Global_Sales")
plt.show()


# We are going to see sales based on rating,
#    -> excluded the rating category with too little data.
df3 = df2[(df2.Rating =='E')|(df2.Rating =='E10+')|(df2.Rating =='T')|(df2.Rating =='M')]

print("==========================")
print("new_Rating")
print(df3["Rating"].value_counts())
print()


#droped Global sales over 20
df3 = df3[df3.Global_Sales < 20]



# 여기 사이에 encoding / scaling 하면 될듯합니다



# preprocessing result
print("--------Final DataSet shape, dtypes --------")
print(df3.shape)
print(df3.dtypes)
print()
print("------------categorical data  -------------")
categorical = df3.dtypes[df3.dtypes == "object"].index
print(categorical)
print()
print("------final categorical data -------")
print(df3[categorical].describe())
print()
print("---final dataset --------")
print(df3)
print()
#release final dataset
df3.to_csv('cleaning_vgsales.csv', index=False, header=True)


