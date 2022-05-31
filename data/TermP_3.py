import numpy as np
import pandas as pd
from numpy import newaxis
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier


#hyperparameter가 존재하므로 evaluation과정에서 hyperparameter 처리과정을 거쳐야합니다.
#regeression
'''
input : traing data(y is target column, X is other columns)
output : trained model
definition : enter the data frame and make the result of trained model which is the
            Gradient Boosting Regressor
'''
def GBR(X,y):

    regressor = GradientBoostingRegressor(
        max_depth=5,
        n_estimators=100,
        learning_rate=0.1
    )
    regressor = regressor.fit(X, y)
    return regressor

#역시 hyperparameter가 존재하므로 evaluation과정에서 hyperparameter처리과정을 거쳐야 합니다.
#classification
'''
input: target(IsSuccess column) and other column for traing and the data that we want to predict
output: result of the KNN
definition : KNN doesn't need the traing so the result of the KNN is directly out
'''
def KNN(X,y, vipX):
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X,y)
    return classifier.predict(vipX)


#시험대상이 될 테스트 데이터입니다.(밑에서도 변수에 vip들어가는건 다 predict의 대상이 되는 data라고 생각하시면 됩니다.)
vip = pd.DataFrame({"Global_Sales": np.nan, "Critic_Score": 87.0, "User_Score":np.nan, "Rating":'M', "Publisher":'Nintendo', "Genre":'Puzzle'}, index=[0])
print("start of vip: ")
print(vip)

#테스팅- 알고리즘이 잘 작동하는지 확인
df = pd.read_csv("cleaning_vgsales.csv")

#encoding이 필요했기에 간단하게 label encoder로 구현해서 테스팅 진행했습니다.(이때 predict에 돌릴데이터도 같이 encoding해줍니다)
le = LabelEncoder()

#장르 인코딩
le.fit(df['Genre'])
newGenre = pd.DataFrame(le.transform(df['Genre']), columns= ["Genre"], dtype=int)

#vip
newVipGenre = pd.DataFrame(le.transform(vip['Genre']), columns=["Genre"], dtype=int)

#게임 제조사 인코딩
le.fit(df['Publisher'])
newPublisher = pd.DataFrame(le.transform(df['Publisher']), columns = ["Publisher"], dtype = int)

#vip
newVipPublisher = pd.DataFrame(le.transform(vip['Publisher']), columns=["Publisher"], dtype=int)

#기존에 있던 categorical 값들 제거하고 인코딩된 형태로 다시 붙여서 새로운 df를 생성했습니다.(vip도 인코딩된것을 잘 붙여줍니다.)
dfNew = df.drop(columns=["Genre","Publisher"])
dfNew["Publisher"] = newPublisher
dfNew["Genre"] = newGenre
dfUser_Score = dfNew.drop(columns=["Global_Sales"])

#vip
vipNew = vip.drop(columns=["Genre","Publisher","Global_Sales"])
vipNew["Publisher"] = newVipPublisher
vipNew["Genre"] = newVipGenre

#인코딩 결과를 확인가능합니다.
print("result of the concatenate encoding data: ")
print(dfNew)
print("this is vip(encoding): ")
print(vipNew)

#run the model
X = dfUser_Score.drop(columns=["User_Score","Rating"])
y = dfUser_Score["User_Score"]

vipX = vipNew.drop(columns=["User_Score","Rating"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2,
                                                        random_state=1,
                                                        shuffle=True)
reg = GBR(X_train,y_train)

#점수가 완전 똥으로 나옵니다 : 점수격차가 너무 큽니다. 따라서 scaling과 evaluation의 hyperparameter처리 과정을 통해서 score를 상승시켜야 합니다.
print("score of the predict User_score: ")
print(reg.score(X_test, y_test))

#여기다가 원하는 값을 집어 넣으면 예측이 됩니다.(평가 점수 예측)
print("##prediction of the User_Score of the vip: ")
print(reg.predict(vipX))

#vip의 User_Score를 세팅해줍니다.
vipNew["User_Score"] = reg.predict(vipX)
print("current vip status(No Global_Sales yet) : ")
print(vipNew)

#based on the User_socre traing and predict the Global sales.
X = dfNew[["User_Score"]]
y = dfNew["Global_Sales"]
vipX = vipNew[["User_Score"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2,
                                                        random_state=1,
                                                        shuffle=True)
reg = GBR(X_train,y_train)

#역시 점수가 완전 똥으로 나옵니다, 역시 판매량 격차가 너무 큽니다.(0.01~16.27) 따라서 scaling과 evaluation의 hyperparameter처리 과정을 통해서 score를 상승시켜야 합니다.
print("score of predict Global_Sales: ")
print(reg.score(X_test, y_test))

#게임의 평가점수를 기반으로 판매량 예측
print("##prediction of Global sales of vip: ")
print(reg.predict(vipX))

#vip의 Global_Sales를 세팅해줍니다
vipNew["Global_Sales"] = reg.predict(vipX)
print("current vip status: ")
vipNew = vipNew[['Global_Sales', 'Critic_Score','User_Score','Rating', 'Publisher', 'Genre']]
print(vipNew)

#add the IsSuccess colunm for predict the game is whether success or not
#여기서 cutoff가 나옵니다. 일단 판매량 평균의 2.0정도를 기준으로 잡았습니다(그러면 1.4정도가 나옵니다.)
cutoff = dfNew["Global_Sales"].mean()*2.0

#cutOff보다 global sales가 큰값에는 success(성공한 게임), 작은 값에는 failure(실패한 게임)을 설정해줍니다
dfNew['IsSuccess'] = ["success" if s >= cutoff else "failure" for s in dfNew['Global_Sales']]
print(dfNew)

#알고리즘을 돌립니다
X = dfNew[["Global_Sales","User_Score"]]
y = dfNew["IsSuccess"]
#vip
vipX = vipNew[["Global_Sales","User_Score"]]

#score를 뽑아줍니다.(KNN의 평가는 confusion matrix 쓰시면 됩니다.)
#

#성공 가능 여부를 vip에 넣어줍니다.
vipNew["IsSuccess"] = KNN(X,y,vipX)
print("vip will be ",end="")
print(vipNew["IsSuccess"].values)
