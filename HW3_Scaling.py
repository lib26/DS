import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler



# Effect of the MinMaxScaler

np.random.seed(1)
df = pd.DataFrame({
    'x1': np.random.normal(0, 2, 10000),
    'x2': np.random.normal(5, 3, 10000),
    'x3': np.random.normal(-5, 5, 10000)
})

np.random.normal(loc = 0.0, scale = 1.0, size = None)

Scaler = MinMaxScaler()
scaled_df = Scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_df, columns=['x1', 'x2', 'x3'])

fig, (ax1, ax2) = plt.subplots(ncols =2, figsize =(6,5))

ax1.set_title('Before Scaling')
sns.kdeplot(df['x1'], ax=ax1)
sns.kdeplot(df['x2'], ax=ax1)
sns.kdeplot(df['x3'], ax=ax1)
ax2.set_title('After MinMaxScaler')
sns.kdeplot(scaled_df['x1'], ax=ax2)
sns.kdeplot(scaled_df['x2'], ax=ax2)
sns.kdeplot(scaled_df['x3'], ax=ax2)
plt.show()



# Effect of the RobustScaler

np.random.seed(1)
df = pd.DataFrame({
    'x1': np.random.normal(0, 2, 10000),
    'x2': np.random.normal(5, 3, 10000),
    'x3': np.random.normal(-5, 5, 10000)
})
np.random.normal(loc = 0.0, scale = 1.0, size = None)

Scaler = RobustScaler()
robustScaled = Scaler.fit_transform(df)
robustScaled = pd.DataFrame(robustScaled, columns=['x1','x2','x3'])

fig, (ax1, ax2) = plt.subplots(ncols =2, figsize =(6,5))

ax1.set_title('Before Scaling')
sns.kdeplot(df['x1'], ax=ax1)
sns.kdeplot(df['x2'], ax=ax1)
sns.kdeplot(df['x3'], ax=ax1)
ax2.set_title('After RobustScaler')
sns.kdeplot(robustScaled['x1'], ax=ax2)
sns.kdeplot(robustScaled['x2'], ax=ax2)
sns.kdeplot(robustScaled['x3'], ax=ax2)
plt.show()




# In-Class Exercise: Standard Scaling

scores=[28, 35, 26, 32, 28, 28, 35, 34, 46, 42, 37]

print("Mean :", '%.2f' %np.mean(scores))
print("Standard Deviation :", '%.2f' %np.std(scores))

mean = np.mean(scores)
std = np.std(scores)

normalized_scores =[]
for value in scores:
    normalized_num = (value - mean)/std
    normalized_scores.append(round(normalized_num, 2))
print("Standard Scores :", normalized_scores)

f_num=[]
cnt=0
for value in np.array(normalized_scores):
    if value<=-1.0:
        f_num.append(cnt)
    cnt+=1

f_scores =[]
for value in f_num:
    f_scores.append(scores[value])

print("F scores :", f_scores)



