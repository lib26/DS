import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
# git test5
# enter the data set
trainDf = pd.read_csv("data/mnist_test.csv")

print("train data set : ")
print(trainDf)

#Using the k-nearest neighbors model
X = trainDf.drop(columns = ['label'])
y = trainDf['label'].values

# KNN, K = 3
knn3 = KNeighborsClassifier(n_neighbors=3)
knn3 = knn3.fit(X,y)

# 5-fold cross validation method for evaluation
cv_scores_3_before = cross_val_score(knn3, X, y, cv=5, verbose = 2)

# print each cv score (accuracy) and average them  print(cv_scores)
print('cv_scores mean(K=3):{}'.format(np.mean(cv_scores_3_before)))

# KNN, K = 5
knn5 = KNeighborsClassifier(n_neighbors=5)
knn5 = knn5.fit(X,y)

# 5-fold cross validation method for evaluation
cv_scores_5_before = cross_val_score(knn5, X, y, cv=5, verbose = 2)

# print each cv score (accuracy) and average them  print(cv_scores)
print('cv_scores mean(K=5):{}'.format(np.mean(cv_scores_5_before)))

# hyerparameter tuning by using GridSeach
#create a dictionary of all values we want to test for n_neighbors
param_grid = {'n_neighbors': np.arrange(1,25)}

#use GridSearch to test all values for n_neighbors(KNN's K = 3)
knn_gscv3 = GridSearchCV(knn3,param_grid, cv = 5)
knn_gscv3 = knn_gscv3.fit(X,y)

#compare with the cv_scores_3_before
# 5-fold cross validation method for evaluation
cv_scores_3_gscv = cross_val_score(knn_gscv3, X, y, cv=5, verbose = 2)

# print each cv score (accuracy) and average them  print(cv_scores)
print("score before gscv(K=3): ")
print(cv_scores_3_before)
print('cv_scores mean_after_gscv(K=3):{}'.format(np.mean(cv_scores_3_gscv)))

#use GridSearch to test all values for n_neighbors(KNN's k = 5)
knn_gscv5 = GridSearchCV(knn5,param_grid, cv = 5)
knn_gscv5 = knn_gscv5.fit(X,y)

#compare with the cv_scores_5_before
# 5-fold cross validation method for evaluation
cv_scores_5_gscv = cross_val_score(knn_gscv5, X, y, cv=5, verbose = 2)

# print each cv score (accuracy) and average them  print(cv_scores)
print("score before gscv(K=5): ")
print(cv_scores_5_before)
print('cv_scores mean_after_gscv(K=5):{}'.format(np.mean(cv_scores_5_gscv)))

# hyerparameter tuning by using RandomizedSeachCV

#use RandomizeSearchCV to test values for n_neighbors(KNN's K = 3)
knn_rscv3 = RandomizedSearchCV(knn3,param_grid, cv = 5)
knn_rscv3 = knn_rscv3.fit(X,y)

#compare with the cv_scores_3_before
# 5-fold cross validation method for evaluation
cv_scores_3_rscv = cross_val_score(knn_rscv3, X, y, cv=5, verbose = 2)

# print each cv score (accuracy) and average them  print(cv_scores)
print("score before rscv(K=3): ")
print(cv_scores_3_before)
print('cv_scores mean_after_rscv(K=3):{}'.format(np.mean(cv_scores_3_rscv)))

#use RandomizeSearchCV to test all values for n_neighbors(KNN's k = 5)
knn_rscv5 = RandomizedSearchCV(knn5,param_grid, cv = 5)
knn_rscv5 = knn_rscv5.fit(X,y)

#compare with the cv_scores_5_before
# 5-fold cross validation method for evaluation
cv_scores_5_rscv = cross_val_score(knn_rscv5, X, y, cv=5, verbose = 2)

# print each cv score (accuracy) and average them  print(cv_scores)
print("score before rscv(K=5): ")
print(cv_scores_5_before)
print('cv_scores mean_after_rscv(K=5):{}'.format(np.mean(cv_scores_5_rscv)))

#compare basic KNN, GSCV KNN, RSCV KNN
print("this is the result for KNN, which K = 3 : ")
print("basic : ",end ="")
print(cv_scores_3_before)
print("after GSCV : ",end="")
print(cv_scores_3_gscv)
print("after RSCV : ",end="")
print(cv_scores_3_rscv)

print("this is the result for KNN, which K = 5 : ")
print("basic : ",end ="")
print(cv_scores_5_before)
print("after GSCV : ",end="")
print(cv_scores_5_gscv)
print("after RSCV : ",end="")
print(cv_scores_5_rscv)
