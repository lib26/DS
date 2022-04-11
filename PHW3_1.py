import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model
import seaborn as sns
from sklearn import preprocessing

#read from a csv file
df = pd.read_excel('data/bmi_data_phw3.xlsx')

#Data exploration
print("#Print dataset statistical data")
print(df.describe())
print()

print("#Print dataset feature names")
print(df.columns.values)
print()

print("#Print dataset feature data types")
print(df.dtypes)

#Plot height & weight histograms (bins=10) for each BMI value
#height histograms
g = sns.FacetGrid(df, col='BMI')
g.map(plt.hist, 'Height (Inches)', bins=10)
plt.show()

#weight histograms
g = sns.FacetGrid(df, col='BMI')
g.map(plt.hist, 'Weight (Pounds)', bins=10)
plt.show()

#Plot scaling results for height and weight
height_weight_Data = df.loc[:, ['Height (Inches)', 'Weight (Pounds)']]

#Standard scaling
stdscaler = preprocessing.StandardScaler()
scaled_df = stdscaler.fit_transform(height_weight_Data)
scaled_df = pd.DataFrame(scaled_df, columns=['Height (Inches)', 'Weight (Pounds)'])

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6,5))

ax1.set_title('Before Scaling')
sns.kdeplot(df['Height (Inches)'], ax=ax1)
sns.kdeplot(df['Weight (Pounds)'], ax=ax1)

ax2.set_title('After Standard Scaler')
sns.kdeplot(scaled_df['Height (Inches)'], ax=ax2)
sns.kdeplot(scaled_df['Weight (Pounds)'], ax=ax2)
plt.show()


#MinMaxScaler
mnscaler = preprocessing.MinMaxScaler()
scaled_df = mnscaler.fit_transform(height_weight_Data)
scaled_df = pd.DataFrame(scaled_df, columns=['Height (Inches)', 'Weight (Pounds)'])

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6,5))

ax1.set_title('Before Scaling')
sns.kdeplot(df['Height (Inches)'], ax=ax1)
sns.kdeplot(df['Weight (Pounds)'], ax=ax1)

ax2.set_title('After MinMax Scaler')
sns.kdeplot(scaled_df['Height (Inches)'], ax=ax2)
sns.kdeplot(scaled_df['Weight (Pounds)'], ax=ax2)
plt.show()


#RobustScaler
rbscaler = preprocessing.RobustScaler()
scaled_df = rbscaler.fit_transform(height_weight_Data)
scaled_df = pd.DataFrame(scaled_df, columns=['Height (Inches)', 'Weight (Pounds)'])

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6,5))

ax1.set_title('Before Scaling')
sns.kdeplot(df['Height (Inches)'], ax=ax1)
sns.kdeplot(df['Weight (Pounds)'], ax=ax1)

ax2.set_title('After Robust Scaler')
sns.kdeplot(scaled_df['Height (Inches)'], ax=ax2)
sns.kdeplot(scaled_df['Weight (Pounds)'], ax=ax2)
plt.show()
