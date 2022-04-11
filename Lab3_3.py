import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model

# https://hleecaster.com/ml-linear-regression-example/ 좋은 참고자료

df = pd.read_csv('data/bmi_data_lab3.csv')

missing = ['-104.4205547', '592.6244266', '664.4877548',
           '622.0486612','665.4650594','-130.9261617',
           '1110.621115', '-161.9949135', '-141.8241248', '0']

df4 = pd.read_csv("data/bmi_data_lab3.csv", na_values=missing).dropna(axis=0)
print(df4) # na 값이 들어있는 행은 다 빠져있음.

# Compute the linear regression equation E for (height, weight) values
weight = np.array(df4['Weight (Pounds)'])
height = np.array(df4['Height (Inches)'])

reg = linear_model.LinearRegression()
reg.fit(height[:, np.newaxis], weight) # height.reshape(-1,1) 랑 같은 말.
# X는 2차원 array 형태여야 하기 때문 이유는 X 변수가 하나가 아니라 여러개일 때 다중회귀분석을 실시하기 위함인데

# fit() 메서드는 선형 회귀 모델에 필요한 두 가지 변수를 전달하는 거다.
#
# 기울기: line_fitter.coef_
# 절편: line_fitter.intercept_
# 어쨌든 이게 끝이다. 이렇게 하면 새로운 X 값을 넣어 y값을 예측할 수 있게 된다.

px = np.array([height.min() - 1, height.max() + 1])
py = reg.predict(px[:, np.newaxis])

plt.title("#Compute linear regression equation")
plt.xlabel('Height (inches)')
plt.ylabel('Weight (pounds)')

plt.scatter(height, weight)
plt.plot(px, py, color='r')
plt.show()

# For dirty height and weight values, compute replacement values using E
dh = []
dw = []
# y(몸무게) = ax + b
# x(키) = (y-b)/a
k = float(reg.coef_) # 기울기
y0 = reg.intercept_ # 절편

# X 축은 키(Height)
# Y 축은 몸무게(Weight)
for i, row in df.copy().dropna().iterrows(): # df.copy().dropna 여기서 빈칸은 NaN 처리 돼서 그 행들은 다 빠짐
    if row['Height (Inches)'] <= 0 or row['Height (Inches)'] > 100: # 키 값이 이상하면
        row['Height (Inches)'] = (row['Weight (Pounds)']-y0)/k # 주어진 y값(=몸무게)을 통해서 x값을 구해라
        dh.append(row['Height (Inches)'])
        dw.append(row['Weight (Pounds)'])
    if row['Weight (Pounds)'] <= 0 or row['Weight (Pounds)'] > 400:
        row['Weight (Pounds)'] = k*(row['Height (Inches)']) + y0
        dh.append(row['Height (Inches)'])
        dw.append(row['Weight (Pounds)'])

plt.title("#Compute replacement values using E")
plt.xlabel('Height (inches)')
plt.ylabel('Weight (pounds)')
plt.scatter(height, weight)
plt.scatter(dh, dw, color='r')
plt.plot(px, py, color='k')
plt.show()


#Do the same for the groups divided by gender and BMI
# Male
MaleData = df4.loc[df4['Sex'] == 'Male'] # df4 갖다 쓰는 거 인지

h = MaleData['Height (Inches)']
w = MaleData['Weight (Pounds)']

reg = linear_model.LinearRegression()
reg.fit(height[:, np.newaxis], weight)
px = np.array([height.min() - 1, height.max() + 1])
py = reg.predict(px[:, np.newaxis])

data = df.copy().dropna().loc[df['Sex'] == 'Male']
dh = []
dw = []
k = float(reg.coef_)
y0 = reg.intercept_

for i, row in data.iterrows():
    if row['Height (Inches)'] <= 0 or row['Height (Inches)'] > 100:
        row['Height (Inches)'] = (row['Weight (Pounds)']-y0)/k
        dh.append(row['Height (Inches)'])
        dw.append(row['Weight (Pounds)'])
    if row['Weight (Pounds)'] <= 0 or row['Weight (Pounds)'] > 400:
        row['Weight (Pounds)'] = k*(row['Height (Inches)']) + y0
        dh.append(row['Height (Inches)'])
        dw.append(row['Weight (Pounds)'])


plt.title("Male")
plt.xlabel('Height (inches)')
plt.ylabel('Weight (pounds)')

plt.scatter(height, weight)
plt.scatter(dh, dw, color='r')
plt.plot(px, py, color='k')
plt.show()


# Female
FemaleData = df4.loc[df4['Sex'] == 'Female']

reg = linear_model.LinearRegression()
reg.fit(height[:, np.newaxis], weight)
px = np.array([height.min() - 1, height.max() + 1])
py = reg.predict(px[:, np.newaxis])

data = df.copy().dropna().loc[df['Sex'] == 'Female']
dh = []
dw = []
k = float(reg.coef_)
y0 = reg.intercept_

for i, row in data.iterrows():
    if row['Height (Inches)'] <= 0 or row['Height (Inches)'] > 100:
        row['Height (Inches)'] = (row['Weight (Pounds)']-y0)/k
        dh.append(row['Height (Inches)'])
        dw.append(row['Weight (Pounds)'])
    if row['Weight (Pounds)'] <= 0 or row['Weight (Pounds)'] > 400:
        row['Weight (Pounds)'] = k*(row['Height (Inches)']) + y0
        dh.append(row['Height (Inches)'])
        dw.append(row['Weight (Pounds)'])

plt.title("Female")
plt.xlabel('Height (inches)')
plt.ylabel('Weight (pounds)')

plt.scatter(height, weight)
plt.scatter(dh, dw, color='r')
plt.plot(px, py, color='k')
plt.show()

# BMI
bmi1Data = df4[df4['BMI'] == 1]
bmi2Data = df4[df4['BMI'] == 2]
bmi3Data = df4[df4['BMI'] == 3]

#weak
height = bmi1Data['Height (Inches)']  # type : Series
weight = bmi1Data['Weight (Pounds)']

reg = linear_model.LinearRegression()
reg.fit(height.values.reshape(-1, 1), weight)
px = np.array([height.min() - 1, height.max() + 1])
py = reg.predict(px[:, np.newaxis])

dh = []
dw = []
k = float(reg.coef_)
y0 = reg.intercept_
data = df4.copy().dropna().loc[df['BMI'] == 1.0]

for i, row in data.iterrows():
    if row['Height (Inches)'] <= 0 or row['Height (Inches)'] > 100:
        row['Height (Inches)'] = (row['Weight (Pounds)']-y0)/k
        dh.append(row['Height (Inches)'])
        dw.append(row['Weight (Pounds)'])
    if row['Weight (Pounds)'] <= 0 or row['Weight (Pounds)'] > 400:
        row['Weight (Pounds)'] = k*(row['Height (Inches)']) + y0
        dh.append(row['Height (Inches)'])
        dw.append(row['Weight (Pounds)'])

plt.title("BMI==1 (weak)")
plt.xlabel('Height (inches)')
plt.ylabel('Weight (pounds)')

plt.scatter(height, weight)
plt.scatter(dh, dw, color='r')
plt.plot(px, py, color='k')
plt.show()


#Normal
height = bmi2Data['Height (Inches)']
weight = bmi2Data['Weight (Pounds)']

reg = linear_model.LinearRegression()
reg.fit(height.values.reshape(-1, 1), weight)
px = np.array([height.min() - 1, height.max() + 1])
py = reg.predict(px[:, np.newaxis])

dh = []
dw = []
k = float(reg.coef_)    # 기울기
y0 = reg.intercept_      # y 절편
data = df4.copy().dropna().loc[df['BMI'] == 2.0]

for i, row in data.iterrows():
    if row['Height (Inches)'] <= 0 or row['Height (Inches)'] > 100:
        row['Height (Inches)'] = (row['Weight (Pounds)']-y0)/k
        dh.append(row['Height (Inches)'])
        dw.append(row['Weight (Pounds)'])
    if row['Weight (Pounds)'] <= 0 or row['Weight (Pounds)'] > 400:
        row['Weight (Pounds)'] = k*(row['Height (Inches)']) + y0
        dh.append(row['Height (Inches)'])
        dw.append(row['Weight (Pounds)'])

plt.title("BMI==2 (normal)")
plt.xlabel('Height (inches)')
plt.ylabel('Weight (pounds)')

plt.scatter(height, weight)
plt.scatter(dh, dw, color='r')
plt.plot(px, py, color='k')
plt.show()


#Overweight
height = bmi3Data['Height (Inches)']
weight = bmi3Data['Weight (Pounds)']

reg = linear_model.LinearRegression()
reg.fit(height.values.reshape(-1, 1), weight)
px = np.array([height.min() - 1, height.max() + 1])
py = reg.predict(px[:, np.newaxis])

dh = []
dw = []
k = float(reg.coef_)    # 기울기
y0 = reg.intercept_      # y 절편
data = df4.copy().dropna().loc[df['BMI'] == 3.0]

for i, row in data.iterrows():
    if row['Height (Inches)'] <= 0 or row['Height (Inches)'] > 100:
        row['Height (Inches)'] = (row['Weight (Pounds)']-y0)/k
        dh.append(row['Height (Inches)'])
        dw.append(row['Weight (Pounds)'])
    if row['Weight (Pounds)'] <= 0 or row['Weight (Pounds)'] > 400:
        row['Weight (Pounds)'] = k*(row['Height (Inches)']) + y0
        dh.append(row['Height (Inches)'])
        dw.append(row['Weight (Pounds)'])

plt.title("BMI==3 (overweight)")
plt.xlabel('Height (inches)')
plt.ylabel('Weight (pounds)')

plt.scatter(height, weight)
plt.scatter(dh, dw, color='r')
plt.plot(px, py, color='k')
plt.show()
