import warnings
warnings.filterwarnings(action="ignore")
from pandas import DataFrame
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import csv
with open('data/winequality-red.csv', 'r') as f:
    csvreader = csv.reader(f, delimiter=";")
    i = 0
    for row in csvreader:
        if(i==0):
            data = DataFrame(columns=row)
            i = i+1
        else:
            data.loc[len(data)] = row

print("<--------------statistical data-------------->")
print(data.describe())
print()
print("<--------------data type-------------->")
print(data.dtypes)
print()


# Encoding string data to num label
enc = LabelEncoder()

for column in data.columns:
    data[column] = enc.fit_transform(data[column])

X = data.drop(columns=['quality'])
y = data.loc[:, ['quality']]


# Split data into 9:1 (training:test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=1000)
# Create decision tree classfier
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(X_train, y_train)
y_prediction = clf.predict(X_test)
# Result
print('9:1(training:test) Accuracy is', accuracy_score(y_test, y_prediction))


# Split data into 8:2 (training:test)
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, shuffle=True, random_state=1000)
# Create decision tree classfier
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(X_train, y_train)
y_prediction = clf.predict(X_test)
# Result
print('8:2(training:test) Accuracy is', accuracy_score(y_test, y_prediction))


# Split data into 7:3 (training:test)
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3, shuffle=True, random_state=1000)
# Create decision tree classfier
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(X_train, y_train)
y_prediction = clf.predict(X_test)
# Result
print('7:3(training:test) Accuracy is', accuracy_score(y_test, y_prediction))
