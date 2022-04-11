import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model
import seaborn as sns
from sklearn import preprocessing

# Find outlier people
# Read the Excel dataset file,
df = pd.read_excel('data/bmi_data_phw3.xlsx')

#Compute the linear regression equation E for the input dataset

height = np.array(df['Height (Inches)'])
weight = np.array(df['Weight (Pounds)'])

reg = linear_model.LinearRegression()
reg.fit(height[:, np.newaxis], weight)

px = np.array([height.min() - 1, height.max() + 1]) # type : numpy.ndarray
py = reg.predict(px[:, np.newaxis]) # type : numpy.ndarray

#Compute linear regression equation
plt.title("Data of all")
plt.xlabel('Height (inches)')
plt.ylabel('Weight (pounds)')

plt.scatter(height, weight)
plt.plot(px, py, color='k')
plt.show()


#Compute e=w-w' (w’ is obtained for h using E)
df_e = df.copy()
df_e['ww'] = np.NaN
e = []

for i in range(len(df_e)): #predict weight by height and store in dataframe
     df_e.loc[i, ['ww']] = reg.predict(height[i].reshape(-1, 1))  # ww = w'

wp = df_e['ww']

for i in range(len(df_e)): #compute e=w-w' than store
    e.append(weight[i]-wp[i])

df_e['Weight (Pounds)'] = e
print('================== df_e ==================')
print(df_e)
print()

#Normalize the e values
mean = np.mean(e) #mean
std = np.std(e) #standard deviation

df_z = df.copy()
df_z['z'] = np.NaN
z = []

for i in range(len(df_e)):
    zz = (df_e['Weight (Pounds)'][i] - mean) / std
    df_z.loc[i, ['z']] = zz
    z.append(zz)

df_z['Weight (Pounds)'] = z
print('================== df_z ==================')
print(df_z)

#Plot a histogram showing the distribution of z
plt.title('Data of all')
plt.hist(z, rwidth=0.8, bins=10)
plt.show()

#Decide a value α
alpha = 1
df_z.loc[df_z['Weight (Pounds)'] < -alpha, 'BMI'] = 0
df_z.loc[df_z['Weight (Pounds)'] > alpha, 'BMI'] = 4

