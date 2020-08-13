import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
data=pd.read_csv('train.csv')
data_test=pd.read_csv('test.csv')
data2=data_test[['condition','color_type','length(m)','height(cm)','X1','X2']]

data1=data[['condition','color_type','length(m)','height(cm)','X1','X2','breed_category','pet_category']]
data1.dropna(axis=0,inplace=True)
data1.isnull()
y=data1[['pet_category']]
y1=data1[['breed_category']]










from sklearn.preprocessing import StandardScaler
SC=StandardScaler()

from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()

data1[['color_type']]=LE.fit_transform(data1[['color_type']])
data1.isnull()
data2[['color_type']]=LE.fit_transform(data2[['color_type']])
data1.drop(columns='pet_category',inplace=True)
data1.drop(columns='breed_category',inplace=True)
data2.isnull()
data2['condition'].replace(np.nan,data2['condition'].median(),inplace=True)
data2.isnull()
#z=pd.get_dummies(data1,columns=['color_type'])
#z.isnull()
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(data1,y,test_size=0.3,random_state=0)

from sklearn.decomposition import PCA
pca=PCA(n_components=None)
#x_train=pca.fit_transform(x_train)
#x_test=pca.fit_transform(x_test)
#variance=pca.explained_variance_
#variance

#y=z[['pet_category']]
#z.drop(columns='pet_category',inplace=True)
#z=SC.fit_transform(z)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
KNN=KNeighborsClassifier()
GNB=GaussianNB()
svc=SVC()
LR=LogisticRegression()
RFC=RandomForestClassifier()
RFC.fit(x_train,y_train)
KNN.fit(x_train,y_train)
LR.fit(x_train,y_train)
pred=RFC.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy_score(pred,y_test)




y_train=np.squeeze(y_train)
y_test=np.squeeze(y_test)

y_train=LE.fit_transform(y_train)
y_test=LE.fit_transform(y_test)



import keras
from keras.layers import Dense,LSTM,Dropout
from keras.models import Sequential
model=Sequential()
model.add(Dense(int(1200),activation='relu',kernel_initializer='normal',input_dim=6))
model.add(Dense(int(100),activation=keras.layers.LeakyReLU(alpha=0.3)))
model.add(Dense(int(2096),activation=keras.layers.LeakyReLU(alpha=0.3)))
#model.add(Dropout(0.2))
model.add(Dense(int(10),activation='relu'))

model.add(Dense(4,activation='softmax'))
#model=Sequential(Dense(int(100),activation='tanh',kernel_initializer='uniform',input_dim=25),Dense(int(1),activation='sigmoid'))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=198)



a=pred2[:,[1]]
b=pred2[:,[2]]
a=a.tolist()
b=b.tolist()
c=[]
for i,j in zip(a,b):
    #c.append(i)
    if i>=j:
        c.append(1)
    elif i<j:
        c.append(2)



pred1=RFC.predict(data2)
pred2=model.predict(data2)














