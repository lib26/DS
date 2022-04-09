import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model

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


