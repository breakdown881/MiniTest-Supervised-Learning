import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from pprint import pprint
import io
import numpy as np
df = pd.read_csv('diabetes.csv')
df.head()
df.columns = ["pregnancies", "glucose", "blood_pressure", "skin_thickness","insulin","bmi","Diabetes_Pedigree_Function","age","outcome"]
df.head()
df.glucose.replace(0,np.nan,inplace = True)
df.insulin.replace(0,np.nan,inplace = True)
df.blood_pressure.replace(0,np.nan,inplace = True)
df.bmi.replace(0,np.nan,inplace = True)
df.skin_thickness.replace(0,np.nan,inplace = True)
df.age.replace(0,np.nan,inplace = True)
df.Diabetes_Pedigree_Function.replace(0,np.nan,inplace = True)
df = df.fillna(df.mean())
df.head()
df.describe()
from sklearn.preprocessing import scale
df['insulin'] = scale(df['insulin'])
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm
y = df['outcome'].values
X = df.drop('outcome',axis =1).values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 42,stratify = y)
import matplotlib.pyplot as plt
import pylab
import numpy as np
neighbors  = np.arange(1,10)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
for i,k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train,y_train)
    train_accuracy[i] = knn.score(X_train,y_train)
    test_accuracy[i] = knn.score(X_test,y_test)
knn = KNeighborsClassifier(n_neighbors = 4)
knn.fit(X_train,y_train)
sample1=[[9,183,65,0,0,22.3,0.692,31]]
sample2=[[2,92,65,24,90,26.1,0.175,20]]
y_pred_1 = knn.predict(sample1)
y_pred_2 = knn.predict(sample2)
print(y_pred_1)
print(y_pred_2)
print(knn.score(X_test,y_test))
clf = svm.SVC()
clf.fit(X_train, y_train)
sample1=[[9,183,65,0,0,22.3,0.692,31]]
sample2=[[2,92,65,24,90,26.1,0.175,20]]
y_pred_1 = knn.predict(sample1)
y_pred_2 = knn.predict(sample2)
print(y_pred_1)
print(y_pred_2)
print(clf.score(X_test, y_test))