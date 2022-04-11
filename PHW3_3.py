import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn import linear_model

# Read the Excel dataset file
df = pd.read_excel('data/bmi_data_phw3.xlsx')

# Create 'male_data' dataset with only male data
male_data = df[df['Sex']=='Male']

height = male_data[['Height (Inches)']] # type : DataFrame
weight = male_data[['Weight (Pounds)']]

# Compute the linear regression
reg = linear_model.LinearRegression()
reg = reg.fit(height,weight)

# Predict weight using height values
px = male_data['Height (Inches)'] # px type : Series
py = reg.predict(male_data[['Height (Inches)']]) #py type : ndarray

# Draw a scatter plot of height, weight
plt.title("Male Data")
plt.xlabel("Height (Inches)")
plt.ylabel("Weight (Pounds)")
plt.scatter(height, weight)
plt.plot(px, py, color='k')
plt.show()

real_weight = weight[['Weight (Pounds)']].to_numpy()  # type : ndarray
predict_weight = py  # type : ndarray

# Compute error values(e=w-w')
error = []
male_data_copy = male_data.copy()
male_data_copy['error'] = np.NaN
male_data_copy['z'] = np.NaN

# Create array that contains error values
for i in range(len(real_weight)):
    error.extend(real_weight[i] - predict_weight[i])
male_data_copy['error'] = error

# Normalize the error values
zscores = stats.zscore(error)
male_data_copy['z'] = zscores

# Plot a histogram showing the distribution of z
plt.hist(zscores, rwidth=0.8, bins=10)
plt.title("Male Result")
plt.xticks([-2, -1, 0, 1, 2])
plt.xlabel('Ze')
plt.ylabel('frequency')
plt.show()

# Decide a value alpha and change values(BMI=0 or BMI=4)
alpha = 1.5
male_data_copy.loc[male_data_copy['z'] < -alpha, 'BMI'] = 0
male_data_copy.loc[male_data_copy['z'] > alpha, 'BMI'] = 4

# Print result
print('================== Original Male Data ==================')
print(male_data)
print('================== Changed Male Data ==================')
print(male_data_copy)
