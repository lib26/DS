import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd

# NumPy
A = [30, 40, 50, 60, 40]
B = [200, 300, 800, 600, 300]
C = [10, 20, 20, 20, 20]
D = [4, 4, 1, 2, 5]

data = np.array([A,B,C,D])

# population covariance matrix (N)
print("============Population===========")
covMatrix = np.cov(data,bias=True)
print(covMatrix)
print()

sn.heatmap(covMatrix, annot=True, fmt='g')
plt.show()

# sample covariance matrix (N-1)
print("============Sample===============")
covMatrix = np.cov(data,bias=False)
print(covMatrix)
print()

sn.heatmap(covMatrix, annot=True, fmt='g')
plt.show()

# Pandas
data = {'Age': [30, 40, 50, 60, 40],
        'Income': [200, 300, 800, 600, 300],
        'Worked': [10, 20, 20, 20, 20],
        'Vacation': [4, 4, 1, 2, 5]}

df = pd.DataFrame(data)
# sample covariance matrix
covMatrix = pd.DataFrame.cov(df)
print(covMatrix)

sn.heatmap(covMatrix, annot=True, fmt='g')
plt.show()
