import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings(action="ignore")

# Read the CSV dataset file & handling NAN data
dataset = pd.read_csv('data/housing.csv')

print("<--------------statistical data-------------->")
print(dataset.describe())
print()
print("<--------------data type-------------->")
print(dataset.dtypes)
print()



dataset.replace({'': np.nan}, inplace=True)
dataset = dataset.dropna(axis=1, how='any')
# dropna & split dataset to target and others
x = dataset.drop(columns=['median_house_value'])
y = dataset['median_house_value']

# Encoder
enc = LabelEncoder()
encoding = pd.DataFrame(dataset['ocean_proximity'])
enc.fit(encoding)
x['ocean_proximity'] = pd.DataFrame(enc.transform(encoding))

# Scaler
Scaler = StandardScaler()
x = Scaler.fit_transform(x)
reg = linear_model.LinearRegression()

# divide into sections
stratify_data = pd.qcut(y, 10)

# Split data into 8:2 (training:test) & shuffle=True & stratify
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, stratify=stratify_data)
reg.fit(x_train, y_train)
predict = reg.predict(x_test)
print("<--------------Split data into 8:2-------------->")
print('Accuracy: ', reg.score(x_test, y_test))
print('Predict: ', predict)
print()

# Split data into 8:2 (training:test) & shuffle=False
x_train2, x_test2, y_train2, y_test2 = train_test_split(x, y, test_size=0.2, shuffle=False)
reg.fit(x_train2, y_train2)
predict2 = reg.predict(x_test2)
print('Accuracy: ', reg.score(x_test2, y_test2))
print('Predict: ', predict2)
print()

# Split data into 6:4 (training:test) & shuffle=True & stratify
x_train3, x_test3, y_train3, y_test3 = train_test_split(x, y, test_size=0.4, shuffle=True, stratify=stratify_data)
reg.fit(x_train3, y_train3)
predict3 = reg.predict(x_test3)
print("<--------------Split data into 6:4-------------->")
print('Accuracy: ', reg.score(x_test3, y_test3))
print('Predict: ', predict3)
print()

# Split data into 6:4 (training:test) & shuffle=False
x_train4, x_test4, y_train4, y_test4 = train_test_split(x, y, test_size=0.4, shuffle=False)
reg.fit(x_train4, y_train4)
predict4 = reg.predict(x_test4)
print('Accuracy: ', reg.score(x_test4, y_test4))
print('Predict: ', predict4)
