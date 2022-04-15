import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

spends = np.array([2400, 3650, 2350, 4950, 3100, 2500, 5106, 3100, 2900, 1750])
spends = spends.reshape(-1, 1)
income = np.array([41200, 50100, 52000, 66000, 44500, 37700, 73500, 37500, 56700, 35600])

# Compute the linear regression equation
reg = linear_model.LinearRegression()
reg.fit(spends, income)
predicted_y = reg.predict(spends)

# plot result
plt.xlabel('spends')
plt.ylabel('income')
plt.scatter(spends, income, color='r')
plt.plot(spends, predicted_y)
plt.show()

# compute m and b
x_total = spends.sum()
y_total = income.sum()
x2_sum=0
for i in range(spends.size):
    x2_sum = x2_sum + spends[i]*spends[i]

xy_sum=0
for i in range(spends.size):
        xy_sum = xy_sum + spends[i] * income[i]

print("x2_sum", x2_sum)
print("xy_sum", xy_sum)
print("x_total", x_total)
print("y_total", y_total)
m = (10*xy_sum-x_total*y_total) / (10*x2_sum-x_total*x_total)
b = (y_total - m * x_total) / 10
print()

print("=========기울기========")
print("m : ", m)
print(reg.coef_)
print()

print("=========기울기========")
print("b : ", b)
print(reg.intercept_)
print()
