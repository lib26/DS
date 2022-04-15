import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import OneHotEncoder

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

# apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(new_X, y)
dfcolumns = pd.DataFrame(new_X.columns)

dfscores = pd.DataFrame(fit.scores_)

# concatenate two dataframes for better visualization
featureScores = pd.concat([dfcolumns, dfscores],axis=1)
featureScores.columns = ['Specs','value']
print(featureScores.nlargest(10,'value'))