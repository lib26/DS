import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model

df = pd.read_csv("data/bmi_data_lab3.csv")

df2 = df.copy()
df2.replace({"": np.nan})
df2.loc[df["Height (Inches)"]<=0, "Height (Inches)"] = np.nan
df2.loc[df["Height (Inches)"]>100, "Height (Inches)"] = np.nan
df2.loc[df["Weight (Pounds)"]<=0, "Weight (Pounds)"] = np.nan
df2.loc[df["Weight (Pounds)"]>400, "Weight (Pounds)"] = np.nan

# 이건 걍 총 개수
print("========= # total NAN =========")
print(df2.isna().values.sum())

# axis 관한 참고자료 : https://hogni.tistory.com/49 /
# array([[8, 6, 9],
#        [0, 1, 5],
#        [7, 7, 8],
#        [8, 4, 1],
#        [2, 3, 2]]) 가 있다고 할 때
# axis = 0 : 여기서 axis = 0은 무엇일까? 5,3 행렬에서 처음 5라고 생각하면되고, 다섯개의 원소를 죄다 더한다고 보면 된다.
# np.sum(array, axis = 0)
# array([25, 21, 25])

# Print # of rows with NAN : nan이 있는 row의 총 개수
print("========= # of rows with NAN =========")
print(df2.isna().any(axis=0).sum())
print(df2.head(20))

print("#Print number of NAN for each column")
print(df2.isna().sum())

print("#All rows without NAN")
print(df2.dropna(axis=0))
print()

#Fill NAN with mean
df3 = df2.copy()
df3['Height (Inches)'].fillna(df['Height (Inches)'].mean(), inplace=True)
df3['Weight (Pounds)'].fillna(df['Weight (Pounds)'].mean(), inplace=True)

#Fill NAN with median
df3 = df2.copy()
df3['Height (Inches)'].fillna(df['Height (Inches)'].median(), inplace=True)
df3['Weight (Pounds)'].fillna(df['Weight (Pounds)'].median(), inplace=True)

#Use ffill
df3 = df2.copy()
df3.fillna(method='ffill', inplace=True)

#Use bfill
df3 = df2.copy()
df3.fillna(method='bfill', inplace=True)




