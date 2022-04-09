import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model

df = pd.read_csv('data/bmi_data_lab3.csv')

missing = ['-104.4205547', '592.6244266', '664.4877548',
           '622.0486612','665.4650594','-130.9261617',
           '1110.621115', '-161.9949135', '-141.8241248', '0']

df4 = pd.read_csv("data/bmi_data_lab3.csv", na_values=missing).dropna(axis=0)

# Compute the linear regression equation E for (height, weight) values
weight = np.array(df4['Weight (Pounds)'])
height = np.array(df4['Height (Inches)'])

reg = linear_model.LinearRegression()
reg.fit(height[:, np.newaxis], weight)

px = np.array([height.min() - 1, height.max() + 1])
py = reg.predict(px[:, np.newaxis])

plt.title("#Compute linear regression equation")
plt.xlabel('Height (inches)')
plt.ylabel('Weight (pounds)')

plt.scatter(height, weight)
plt.plot(px, py, color='k')
plt.show()

# For dirty height and weight values, compute replacement values using E
dh = []
dw = []
k = float(reg.coef_)
y0 = reg.intercept_

for i, row in df.copy().dropna().iterrows():
    if row['Height (Inches)'] <= 0 or row['Height (Inches)'] > 100:
        row['Height (Inches)'] = (row['Weight (Pounds)']-y0)/k
        dh.append(row['Height (Inches)'])
        dw.append(row['Weight (Pounds)'])
    if row['Weight (Pounds)'] <= 0 or row['Weight (Pounds)'] > 400:
        row['Weight (Pounds)'] = k*(row['Height (Inches)']) + y0
        dh.append(row['Height (Inches)'])
        dw.append(row['Weight (Pounds)'])

plt.title("#Compute replacement values using E")
plt.xlabel('Height (inches)')
plt.ylabel('Weight (pounds)')
plt.scatter(height, weight)
plt.scatter(dh, dw, color='r')
plt.plot(px, py, color='k')
plt.show()


#Do the same for the groups divided by gender and BMI
# Male
MaleData = df4.loc[df4['Sex'] == 'Male']

h = MaleData['Height (Inches)']
w = MaleData['Weight (Pounds)']

reg = linear_model.LinearRegression()
reg.fit(height[:, np.newaxis], weight)
px = np.array([height.min() - 1, height.max() + 1])
py = reg.predict(px[:, np.newaxis])

data = df.copy().dropna().loc[df['Sex'] == 'Male']
dh = []
dw = []
k = float(reg.coef_)
y0 = reg.intercept_

for i, row in data.iterrows():
    if row['Height (Inches)'] <= 0 or row['Height (Inches)'] > 100:
        row['Height (Inches)'] = (row['Weight (Pounds)']-y0)/k
        dh.append(row['Height (Inches)'])
        dw.append(row['Weight (Pounds)'])
    if row['Weight (Pounds)'] <= 0 or row['Weight (Pounds)'] > 400:
        row['Weight (Pounds)'] = k*(row['Height (Inches)']) + y0
        dh.append(row['Height (Inches)'])
        dw.append(row['Weight (Pounds)'])


plt.title("Male")
plt.xlabel('Height (inches)')
plt.ylabel('Weight (pounds)')

plt.scatter(height, weight)
plt.scatter(dh, dw, color='r')
plt.plot(px, py, color='k')
plt.show()


# Female
FemaleData = df4.loc[df4['Sex'] == 'Female']

reg = linear_model.LinearRegression()
reg.fit(height[:, np.newaxis], weight)
px = np.array([height.min() - 1, height.max() + 1])
py = reg.predict(px[:, np.newaxis])

data = df.copy().dropna().loc[df['Sex'] == 'Female']
dh = []
dw = []
k = float(reg.coef_)
y0 = reg.intercept_

for i, row in data.iterrows():
    if row['Height (Inches)'] <= 0 or row['Height (Inches)'] > 100:
        row['Height (Inches)'] = (row['Weight (Pounds)']-y0)/k
        dh.append(row['Height (Inches)'])
        dw.append(row['Weight (Pounds)'])
    if row['Weight (Pounds)'] <= 0 or row['Weight (Pounds)'] > 400:
        row['Weight (Pounds)'] = k*(row['Height (Inches)']) + y0
        dh.append(row['Height (Inches)'])
        dw.append(row['Weight (Pounds)'])

plt.title("Female")
plt.xlabel('Height (inches)')
plt.ylabel('Weight (pounds)')

plt.scatter(height, weight)
plt.scatter(dh, dw, color='r')
plt.plot(px, py, color='k')
plt.show()

# BMI
bmi1Data = df4[df4['BMI'] == 1]
bmi2Data = df4[df4['BMI'] == 2]
bmi3Data = df4[df4['BMI'] == 3]

#weak
height = bmi1Data['Height (Inches)']
weight = bmi1Data['Weight (Pounds)']

reg = linear_model.LinearRegression()
reg.fit(height.values.reshape(-1, 1), weight)
px = np.array([height.min() - 1, height.max() + 1])
py = reg.predict(px[:, np.newaxis])

dh = []
dw = []
k = float(reg.coef_)
y0 = reg.intercept_
data = df4.copy().dropna().loc[df['BMI'] == 1.0]

for i, row in data.iterrows():
    if row['Height (Inches)'] <= 0 or row['Height (Inches)'] > 100:
        row['Height (Inches)'] = (row['Weight (Pounds)']-y0)/k
        dh.append(row['Height (Inches)'])
        dw.append(row['Weight (Pounds)'])
    if row['Weight (Pounds)'] <= 0 or row['Weight (Pounds)'] > 400:
        row['Weight (Pounds)'] = k*(row['Height (Inches)']) + y0
        dh.append(row['Height (Inches)'])
        dw.append(row['Weight (Pounds)'])

plt.title("BMI==1 (weak)")
plt.xlabel('Height (inches)')
plt.ylabel('Weight (pounds)')

plt.scatter(height, weight)
plt.scatter(dh, dw, color='r')
plt.plot(px, py, color='k')
plt.show()


#Normal
height = bmi2Data['Height (Inches)']
weight = bmi2Data['Weight (Pounds)']

reg = linear_model.LinearRegression()
reg.fit(height.values.reshape(-1, 1), weight)
px = np.array([height.min() - 1, height.max() + 1])
py = reg.predict(px[:, np.newaxis])

dh = []
dw = []
k = float(reg.coef_)    # 기울기
y0 = reg.intercept_      # y 절편
data = df4.copy().dropna().loc[df['BMI'] == 2.0]

for i, row in data.iterrows():
    if row['Height (Inches)'] <= 0 or row['Height (Inches)'] > 100:
        row['Height (Inches)'] = (row['Weight (Pounds)']-y0)/k
        dh.append(row['Height (Inches)'])
        dw.append(row['Weight (Pounds)'])
    if row['Weight (Pounds)'] <= 0 or row['Weight (Pounds)'] > 400:
        row['Weight (Pounds)'] = k*(row['Height (Inches)']) + y0
        dh.append(row['Height (Inches)'])
        dw.append(row['Weight (Pounds)'])

plt.title("BMI==2 (normal)")
plt.xlabel('Height (inches)')
plt.ylabel('Weight (pounds)')

plt.scatter(height, weight)
plt.scatter(dh, dw, color='r')
plt.plot(px, py, color='k')
plt.show()


#Overweight
height = bmi3Data['Height (Inches)']
weight = bmi3Data['Weight (Pounds)']

reg = linear_model.LinearRegression()
reg.fit(height.values.reshape(-1, 1), weight)
px = np.array([height.min() - 1, height.max() + 1])
py = reg.predict(px[:, np.newaxis])

dh = []
dw = []
k = float(reg.coef_)    # 기울기
y0 = reg.intercept_      # y 절편
data = df4.copy().dropna().loc[df['BMI'] == 3.0]

for i, row in data.iterrows():
    if row['Height (Inches)'] <= 0 or row['Height (Inches)'] > 100:
        row['Height (Inches)'] = (row['Weight (Pounds)']-y0)/k
        dh.append(row['Height (Inches)'])
        dw.append(row['Weight (Pounds)'])
    if row['Weight (Pounds)'] <= 0 or row['Weight (Pounds)'] > 400:
        row['Weight (Pounds)'] = k*(row['Height (Inches)']) + y0
        dh.append(row['Height (Inches)'])
        dw.append(row['Weight (Pounds)'])

plt.title("BMI==3 (overweight)")
plt.xlabel('Height (inches)')
plt.ylabel('Weight (pounds)')

plt.scatter(height, weight)
plt.scatter(dh, dw, color='r')
plt.plot(px, py, color='k')
plt.show()
