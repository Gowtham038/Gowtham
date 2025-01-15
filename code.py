import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
%matplotlib inline

#Load Data From CSV File
df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%206/cbb.csv')
df.head()
df.shape#(rows,columns)
#Add Column
df['windex'] = np.where(df.WAB > 7, 'True', 'False')
df
#Data visualization and pre-processing
df1 = df.loc[df['POSTSEASON'].str.contains('F4|S16|E8', na=False)]
df1.head()
df1['POSTSEASON'].value_counts()
# notice: installing seaborn might takes a few minutes
!conda install -c anaconda seaborn -y
df1
import seaborn as sns

bins = np.linspace(df1.BARTHAG.min(), df1.BARTHAG.max(), 10)
g = sns.FacetGrid(df1, col="windex", hue="POSTSEASON", palette="Set1", col_wrap=6)
g.map(plt.hist, 'BARTHAG', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()

bins = np.linspace(df1.ADJOE.min(), df1.ADJOE.max(), 10)
g = sns.FacetGrid(df1, col="windex", hue="POSTSEASON", palette="Set1", col_wrap=2)
g.map(plt.hist, 'ADJOE', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()
#Pre-processing: Feature selection/extraction

bins = np.linspace(df1.ADJDE.min(), df1.ADJDE.max(), 10)
g = sns.FacetGrid(df1, col="windex", hue="POSTSEASON", palette="Set1", col_wrap=2)
g.map(plt.hist, 'ADJDE', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()

#Convert Categorical features to numerical values
df1.groupby(['windex'])['POSTSEASON'].value_counts(normalize=True)
df1['windex'].replace(to_replace=['False','True'], value=[0,1],inplace=True)
df1.head()
#Feature selection
X = df1[['G', 'W', 'ADJOE', 'ADJDE', 'BARTHAG', 'EFG_O', 'EFG_D',
       'TOR', 'TORD', 'ORB', 'DRB', 'FTR', 'FTRD', '2P_O', '2P_D', '3P_O',
       '3P_D', 'ADJ_T', 'WAB', 'SEED', 'windex']]
X[0:5]
y = df1['POSTSEASON'].values
y[0:5]
#Normalize Data
X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]
#Training and Validation
# We split the X into train and test to find the best k
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Validation set:', X_val.shape,  y_val.shape)
#K Nearest Neighbor(KNN)
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_val)
# Calculate the accuracy
accuracy = accuracy_score(y_val, y_pred)
print(f'Accuracy: {accuracy}')
#Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)
y_pred=dt.predict(X_val)
# Calculate the accuracy
accuracy = accuracy_score(y_val, y_pred)
print(f'Accuracy: {accuracy}')
Support Vector Machine
from sklearn.svm import SVC
svm=SVC()
svm.fit(X_train,y_train)
y_pred=svm.predict(X_val)
# Calculate the accuracy
accuracy = accuracy_score(y_val, y_pred)
print(f'Accuracy: {accuracy}')
Logistic Regression
from sklearn.linear_model import LogisticRegression
log=LogisticRegression(C=0.01)
log.fit(X_train,y_train)
# Calculate the accuracy
accuracy = accuracy_score(y_val, y_pred)
print(f'Accuracy: {accuracy}')
#Model Evaluation using Test set
from sklearn.metrics import f1_score
# for f1_score please set the average parameter to 'micro'
from sklearn.metrics import log_loss
def jaccard_index(predictions, true):
    if (len(predictions) == len(true)):
        intersect = 0;
        for x,y in zip(predictions, true):
            if (x == y):
                intersect += 1
        return intersect / (len(predictions) + len(true) - intersect)
    else:
        return -1
test_df = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0120ENv3/Dataset/ML0101EN_EDX_skill_up/basketball_train.csv',error_bad_lines=False)
test_df.head()
test_df['windex'] = np.where(test_df.WAB > 7, 'True', 'False')
test_df1 = test_df[test_df['POSTSEASON'].str.contains('F4|S16|E8', na=False)]
test_Feature = test_df1[['G', 'W', 'ADJOE', 'ADJDE', 'BARTHAG', 'EFG_O', 'EFG_D',
       'TOR', 'TORD', 'ORB', 'DRB', 'FTR', 'FTRD', '2P_O', '2P_D', '3P_O',
       '3P_D', 'ADJ_T', 'WAB', 'SEED', 'windex']]
test_Feature['windex'].replace(to_replace=['False','True'], value=[0,1],inplace=True)
test_X=test_Feature
test_X= preprocessing.StandardScaler().fit(test_X).transform(test_X)
test_X[0:5]
test_y = test_df1['POSTSEASON'].values
test_y[0:5]
# Assuming k=5 was the best hyperparameter
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(test_X)
f1_knn = f1_score(test_y, y_pred_knn, average='micro')
jaccard_knn = jaccard_index(y_pred_knn, test_y)
print(f'KNN - F1 Score: {f1_knn}, Jaccard Index: {jaccard_knn}')
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(test_X)
f1_dt = f1_score(test_y, y_pred_dt, average='micro')
jaccard_dt = jaccard_index(y_pred_dt, test_y)
print(f'Decision Tree - F1 Score: {f1_dt}, Jaccard Index: {jaccard_dt}')
# Assuming 'rbf' was the best kernel
svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(test_X)
f1_svm = f1_score(test_y, y_pred_svm, average='micro')
jaccard_svm = jaccard_index(y_pred_svm, test_y)
print(f'SVM - F1 Score: {f1_svm}, Jaccard Index: {jaccard_svm}')
log_reg = LogisticRegression(C=0.01)
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(test_X)
f1_log_reg = f1_score(test_y, y_pred_log_reg, average='micro')
jaccard_log_reg = jaccard_index(y_pred_log_reg, test_y)
print(f'Logistic Regression - F1 Score: {f1_log_reg}, Jaccard Index: {jaccard_log_reg}')
