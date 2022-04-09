import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns

data1 = pd.read_csv('data/housing.csv')

# target
y = data1.iloc[:, -2]
# independent
X = data1.iloc[: , 2:8]
# convert categorical value into features
categorical_feature = data1.iloc[: , [-1]]
encoder = OneHotEncoder(sparse=False)
onehot = encoder.fit_transform(categorical_feature)

new_X = pd.concat([y, X], axis=1)
new_X = new_X.fillna(0)


corrmat = new_X.corr()
top_corr_features=corrmat.index
plt.figure(figsize=(20,20))
g=sns.heatmap(data1[top_corr_features].corr(), annot=  True, cmap="RdYlGn")
plt.show()