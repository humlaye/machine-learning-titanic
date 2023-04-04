import numpy as np 
import pandas as pd 

data = pd.read_csv("train.csv")
data_test = pd.read_csv("test.csv")

print(data.isnull().sum())

isimler = ["yolcu_kimligi","hayatta_kaldı","psınıfı","isim","cinsiyet","yaş","sibsp","parch","bilet","ucret","kabin","binis"]
isimleri = ["yolcu_kimligi","psınıfı","isim","cinsiyet","yaş","sibsp","parch","bilet","ucret","kabin","binis"]
data.columns = isimler
data_test.columns = isimleri

print(data.head())

x = data[["psınıfı","cinsiyet","yaş","sibsp","parch"]]
y = data.iloc[:,1]
test = data_test[["psınıfı","cinsiyet","yaş","sibsp","parch"]]
print(x.isnull().sum())

x.iloc[:,2].fillna(x.iloc[:,2].median(), inplace=True)
test.iloc[:,2].fillna(test.iloc[:,2].median(), inplace=True)
print(x.isnull().sum())

from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

le = preprocessing.LabelEncoder()
x.iloc[:,1] = le.fit_transform(x.iloc[:,1])

le1 = preprocessing.LabelEncoder()
test.iloc[:,1] = le1.fit_transform(test.iloc[:,1])
"""
ohe = OneHotEncoder()
x = ohe.fit_transform(x)

ohe1 = OneHotEncoder()
test = ohe1.fit_transform(test)
"""

"""
from sklearn.preprocessing import StandardScaler
sc = StandardScaler(with_mean=(False))
x = sc.fit_transform(x)

y=y.values
test=test.values
from sklearn.svm import SVC

svm = SVC()
svm.fit(x, y)

svm_predict = svm.predict(test)
yolcu_kimlik = data_test[["yolcu_kimligi"]]
yolcu_kimlik.columns=["PassengerId"]
test1 = pd.DataFrame(svm_predict)
test1.columns=["Survived"]
test1 = pd.concat([yolcu_kimlik,test1],axis=1)
test1.to_csv("test1.csv",index=False)
"""
x=x.values
y=y.values
test=test.values

"""
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
dtc.fit(x,y)

from sklearn.model_selection import cross_val_score
cvs = cross_val_score(dtc, X=x,y=y,cv=4)
print(cvs.mean())
print(cvs.std())

from sklearn.model_selection import GridSearchCV
p = [{"criterion":["gini","entropy","log_loss"],"splitter":["best","entropy","random"]}]
grid = GridSearchCV(dtc, param_grid=p,n_jobs=(-1),cv=10)
grid.fit(X=x,y=y)
print("-----------------")
print(grid.best_estimator_)
print(grid.best_score_)

dtc_predict = dtc.predict(test)

yolcu_kimlik = data_test[["yolcu_kimligi"]]
yolcu_kimlik.columns=["PassengerId"]
test2 = pd.DataFrame(dtc_predict)
test2.columns=["Survived"]
test2 = pd.concat([yolcu_kimlik,test2],axis=1)
test2.to_csv("test2.csv",index=False)
"""

"""
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
sc.fit_transform(x)
sc1 = StandardScaler()
sc1.fit_transform(test)
 

from sklearn.linear_model import LinearRegression
lin = LinearRegression()
lin.fit(x, y)
lin_predict = lin.predict(test)

lin_predict = np.round(lin_predict).astype(int)

yolcu_kimlik = data_test[["yolcu_kimligi"]]
yolcu_kimlik.columns=["PassengerId"]
test3 = pd.DataFrame(lin_predict)
test3.columns=["Survived"]
test3 = pd.concat([yolcu_kimlik,test3],axis=1)
test3.to_csv("test3.csv",index=False)
"""

"""
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb = gnb.fit(x,y)
gnb_predict = gnb.predict(test)

yolcu_kimlik = data_test[["yolcu_kimligi"]]
yolcu_kimlik.columns=["PassengerId"]
test4 = pd.DataFrame(gnb_predict)
test4.columns=["Survived"]
test4 = pd.concat([yolcu_kimlik,test4],axis=1)
test4.to_csv("test4.csv",index=False)
"""


from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
sc.fit_transform(x)
sc1 = StandardScaler()
sc1.fit_transform(test)

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(3, kernel_initializer='uniform', activation = 'relu' , input_dim = 5))

classifier.add(Dense(3, kernel_initializer='uniform', activation = 'relu'))

classifier.add(Dense(1, kernel_initializer='uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss =  'binary_crossentropy' , metrics = ['accuracy'] )

classifier.fit(x, y, epochs=50)

y_pred = classifier.predict(test)

y_pred = np.round(y_pred).astype(int)

yolcu_kimlik = data_test[["yolcu_kimligi"]]
yolcu_kimlik.columns=["PassengerId"]
test5 = pd.DataFrame(y_pred)
test5.columns=["Survived"]
test5 = pd.concat([yolcu_kimlik,test5],axis=1)
test5.to_csv("test5.csv",index=False)
