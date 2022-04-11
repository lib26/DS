import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import ExtraTreeClassifier

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

model=ExtraTreeClassifier()
model.fit(new_X,y)
print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=new_X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()