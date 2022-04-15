import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

# input data
data = {'Age': [30, 40, 50, 60, 40],
        'Income': [200, 300, 800, 600, 300],
        'Worked': [10, 20, 20, 20, 20],
        'Vacation': [4, 4, 1, 2, 5]}
df = pd.DataFrame(data)

#compute mean
mean_age = df['Age'].mean()
mean_income = df['Income'].mean()
mean_worked = df['Worked'].mean()
mean_vacation = df['Vacation'].mean()

d_age = data['Age'] - mean_age
d_income = data['Income'] - mean_income
d_worked = data['Worked'] - mean_worked
d_vacation = data['Vacation'] - mean_vacation

#compute D
d = pd.DataFrame(index=['p1','p2','p3','p4','p5'],
                 columns=['d_age', 'd_income', 'd_worked', 'd_vacation'])
d.loc[:, 'd_age'] = d_age
d.loc[:, 'd_income'] = d_income
d.loc[:, 'd_worked'] = d_worked
d.loc[:, 'd_vacation'] = d_vacation

#compute D_Prime
dPrime = np.transpose(d)

#compute covariance matrix
v = dPrime.dot(d)
#print(v)

print("#Population")
print(v/5)
print()
print("#Sample")
print(v/4)
print()

sn.heatmap(v/5, annot=True, fmt='g')
plt.show()

sn.heatmap(v/4, annot=True, fmt='g')
plt.show()