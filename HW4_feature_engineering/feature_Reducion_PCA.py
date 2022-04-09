import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder

data = pd.read_csv('data/housing.csv')

# target
y = data.iloc[:, -2]
# independent
X = data.iloc[: , 2:8]
# convert categorical value into features
categorical_feature = data.iloc[: , [-1]]
encoder = OneHotEncoder(sparse=False)
onehot = encoder.fit_transform(categorical_feature)
categorical_ = pd.DataFrame(encoder.transform(categorical_feature),
                            columns=['bay', '1h', 'ocean', 'inland', 'what'])
new_X = pd.concat([X, categorical_], axis=1)
new_X = new_X.fillna(0)

# scaling
new_X = StandardScaler().fit_transform(new_X)


pca = PCA(5)
principalComponents = pca.fit_transform(new_X)
principalDf = pd.DataFrame(data=principalComponents,
                           columns = ['principal component1', 'principal component2'
                                      , 'principal component3', 'principal component4'
                                      , 'principal component5'])

print(principalDf)
print(y)
print(pd.concat([principalDf,y], axis=1))