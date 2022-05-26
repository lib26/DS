import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

#ignore warning
from sklearn import preprocessing
warnings.filterwarnings('ignore')

df = pd.read_csv("vgsales.csv")

#1. platform - histogram
print("\n--Platform--")
print(df["Platform"].value_counts())

plt.figure(figsize=[13,5])
df["Platform"] = df["Platform"].astype(str)
plt.hist(df["Platform"],bins = 32, rwidth=0.9)
plt.title ("Platform")
plt.xticks(rotation = 40)
plt.xlabel("Platform")
plt.ylabel("The number of game")

##Numerical Data
#2. years - histogram
plt.figure()
plt.hist(df["Year_of_Release"],bins=8, rwidth=0.8)
plt.title ("Years of Release")
plt.xlabel("Years")
plt.ylabel("The number of game")

#3. Genre - histogram
print("--------------Genre----------------")
print(df["Genre"].value_counts())

plt.figure(figsize=[13,5])
df["Genre"] = df["Genre"].astype(str)
plt.hist(df["Genre"],bins = 13, rwidth=0.9)
plt.title ("Genre")
plt.xlabel("Genre")
plt.ylabel("The number of game")

#4. publisher
print("\n--publisher--")
print(df["Publisher"].value_counts())

#5. sales - piechart
plt.figure()
sizes = [df['NA_Sales'].sum(), df['JP_Sales'].sum(), df['EU_Sales'].sum(), df['Other_Sales'].sum()]
labels = "NA_Sales", "JP_Sales", "EU_Sales", "Other_Sales"
plt.pie(sizes, labels=labels, autopct="%1.2f%%")

#6. critical, user - histogram
fig, (ax1,ax2,ax3,ax4) = plt.subplots(ncols=4, figsize=(14,5)) #show 4 graphs in one figure
ax1.set_title('Critic_Score')
sns.distplot(df['Critic_Score'], kde=False,ax=ax1)

ax2.set_title('Critic_Count')
sns.distplot(df['Critic_Count'], kde=False,ax=ax2)

ax3.set_title('User_Score')
df['User_Score'] = np.where(df['User_Score']=='tbd',None,df['User_Score'])
sns.distplot(df['User_Score'], kde=False,ax=ax3)

ax4.set_title('User_Count')
sns.distplot(df['User_Count'], kde=False,ax=ax4)
#show a lot of null

fig, (ax1,ax2,ax3,ax4) = plt.subplots(ncols=4, figsize=(14,5)) #show 4 graphs in one figure
ax1.set_title('Critic_Score')
a = df['Critic_Score'].value_counts().sum()
b = df["Critic_Score"].isnull().sum()
index = ["not null", "null"]
sns.barplot(index, [a,b],tick_label=index,ax=ax1)

ax2.set_title('Critic_Count')
a = df['Critic_Count'].value_counts().sum()
b = df["Critic_Count"].isnull().sum()
index = ["not null", "null"]
sns.barplot(index, [a,b],tick_label=index,ax=ax2)

ax3.set_title('User_Score')
a = df['User_Score'].value_counts().sum()
b = df["User_Score"].isnull().sum()
index = ["not null", "null"]
sns.barplot(index, [a,b],tick_label=index,ax=ax3)

ax4.set_title('User_Count')
a = df['User_Count'].value_counts().sum()
b = df["User_Count"].isnull().sum()
index = ["not null", "null"]
sns.barplot(index, [a,b],tick_label=index,ax=ax4)

#7. developer - histogram
print("\n--Developer--")
print(df["Developer"].value_counts())

#8. rating - pie chart
print("\n-- rating--")
print(df["Rating"].value_counts())
print((df['Rating']=='E').count())
plt.figure()
sizes = [3991,2961,1563,1420,15]
labels = "E", "T", "M", "E10+","etc."
plt.pie(sizes, labels=labels, autopct="%1.2f%%")

plt.show()
