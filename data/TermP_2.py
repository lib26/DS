import numpy as np
import pandas as pd
import pickle
import os
import math
import matplotlib.pyplot as plt
import sns as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler, Normalizer

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


# Data cleaning
# Show all nan values
# As you can see Score, Count, Developer and Rating has the most NAN values
print("-----Number of NAN -------")
print(df.isnull().sum())
print()


# First, discard all 'features' that are not related to Rating and sales volume,
# and then discard rows with NA values.
df2 = df.drop(['Name', 'Platform', 'Year_of_Release', 'Developer',
               'JP_Sales', 'Other_Sales',
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

# 평점은 관련 없으니 drop
df2 = df2.drop(['Critic_Score','User_Score'], axis=1)

# We are going to see sales based on rating,
#    -> excluded the rating category with too little data.
df3 = df2[(df2.Rating =='E')|(df2.Rating =='E10+')|(df2.Rating =='T')|(df2.Rating =='M')]

print("==========================")
print("new_Rating")
print(df3["Rating"].value_counts())
print()

#droped Global sales over 20
df3 = df3[df3.Global_Sales < 20]

# Label Encoder
enc = LabelEncoder()

# 장르를 인코딩함
encoding = pd.DataFrame(df3['Genre'])
enc.fit(encoding)
df3['Genre'] = pd.DataFrame(enc.transform(encoding))

# 제조사를 인코딩함
encoding = pd.DataFrame(df3['Publisher'])
enc.fit(encoding)
df3['Publisher'] = pd.DataFrame(enc.transform(encoding))

# Rating를 인코딩함
encoding = pd.DataFrame(df3['Rating'])
enc.fit(encoding)
df3['Rating'] = pd.DataFrame(enc.transform(encoding))

# 인코딩된 결과를 확인 가능
print("-------- result of the concatenate encoding data --------")
print(df3)

# Standard Scaler
# 각각의 column을 scale하기 위하여 모양을 바꿈
# genre = np.array(df3.loc[:, ['Genre']]).reshape(-1)
# publisher = np.array(df3.loc[:, ['Publisher']]).reshape(-1)
global_sales = np.array(df3.loc[:, ['Global_Sales']]).reshape(-1)
NA_Sales = np.array(df3.loc[:, ['NA_Sales']]).reshape(-1)
EU_Sales = np.array(df3.loc[:, ['EU_Sales']]).reshape(-1)
rating = np.array(df3.loc[:, ['Rating']]).reshape(-1)

Scaler = StandardScaler()
# scaled_genre = Scaler.fit_transform(genre[:, np.newaxis]).reshape(-1)
# scaled_publisher = Scaler.fit_transform(publisher[:, np.newaxis]).reshape(-1)
scaled_global_sales = Scaler.fit_transform(global_sales[:, np.newaxis]).reshape(-1)
scaled_NA_Sales = Scaler.fit_transform(NA_Sales[:, np.newaxis]).reshape(-1)
scaled_EU_Sales = Scaler.fit_transform(EU_Sales[:, np.newaxis]).reshape(-1)
scaled_rating = Scaler.fit_transform(rating[:, np.newaxis]).reshape(-1)

# 새로운 df4에 scaled된 데이터를 넣음
df4 = pd.DataFrame({
    'Genre':df3['Genre'],
    'Publisher':df3['Publisher'],
    'Global_Sales':scaled_global_sales,
    'NA_Sales':scaled_NA_Sales,
    'EU_Sales':scaled_EU_Sales,
    'Rating':scaled_rating
})

cor = df4.corr()
print("------------- corr matrix ----------------")
print(cor)
print()

# rating 상관관계가 없으니 drop
df4 = df4.drop(['Rating'], axis=1)

# scaler 결과를 확인 가능
print("-------- result of the concatenate scaling data --------")
print(df4)

# preprocessing result
print("--------Final DataSet shape, dtypes --------")
print(df4.shape)
print(df4.dtypes)
print()
print("------------categorical data  -------------")
categorical = df4.dtypes[df4.dtypes == "object"].index
print(categorical)
print()

print("---final dataset --------")
print(df4)
print()

# release final dataset
df4.to_csv('cleaning_vgsales.csv', index=False, header=True)
