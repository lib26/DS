import numpy as np
from matplotlib import pyplot as plt
import random

arr = np.arange(4)
r = arr[:,np.newaxis]
print(r)

#[Ex-1]
WT = []
HT = []

for i in range(100):
    weight = random.uniform(40, 90)
    height = random.randint(140, 200) * 0.01
    WT.append(weight)
    HT.append(height)

WT = np.array(WT)
HT = np.array(HT)

BMI = np.array(WT / HT ** 2)
print('[BMI]')
print(BMI[: 10])

#[Ex-2]
#Bar Chart
ws = ['Underweight','Healthy','Overweight','Obese']
uw=healthy=ow=obese = 0

for i in BMI:
    if(i < 18.5):
        uw += 1
    elif(18.5 <= i <25):
        healthy += 1
    elif(25 <= i < 30):
        ow += 1
    elif(i > 30):
        obese += 1

ws_count = [uw,healthy,ow,obese]

plt.bar(ws,ws_count)
plt.title('Bar Chart')
plt.show()

#Histogram
plt.hist(BMI, bins = 4)
plt.xticks([0,18.5,25,30,50])
plt.xlabel('BMI')
plt.ylabel('count of students')
plt.title('Histogram')
plt.show()

#Pie Chart
plt.pie(ws_count, labels = ws, autopct='%1.2f%%')
plt.title('Pie Chart')
plt.show()

#Scatter Plot
plt.scatter(HT*100, WT)
plt.xlabel('Height')
plt.ylabel('Weight')
plt.title('Scatter Plot')
plt.show()

