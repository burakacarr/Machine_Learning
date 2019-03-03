# -*- coding: utf-8 -*-

#veri ve kütüphane yüklenmesi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer,LabelEncoder,OneHotEncoder
    
missingdata = pd.read_csv('data.csv')


imputer = Imputer(missing_values='NaN',strategy='mean', axis=0)
#age weight length = awl

awl = missingdata.iloc[:,1:4].values 

imputer = imputer.fit(awl[:,1:4])
awl[:,1:4] = imputer.transform(awl[:,1:4])

cinsiyet =missingdata.iloc[:,-1].values
print(awl)

ohe=OneHotEncoder(categorical_features='all')
le= LabelEncoder()

nationality = missingdata.iloc[:,0:1].values
print(nationality)
nationality[:,0] = le.fit_transform(nationality[:,0])

print(nationality)
nationality = ohe.fit_transform(nationality).toarray()
print(nationality)



result = pd.DataFrame(data=nationality, index=range(22),columns=["fr","tr","usa"])
print(result)


result2 = pd.DataFrame(data=awl,index=range(22),columns=["boy","kilo","yas"])

print(result2)


result3 = pd.DataFrame(data=cinsiyet,index=range(22),columns=["cinsiyet"])
print(result3)
dataresult= pd.concat([result1,result2,result3],axis=1)

print(dataresult)

from sklearn.cross_validation import train_test_split

x_train,x_test,y_train,y_test= train_test_split(dataresult,result3,test_size=0.33,random_state=0)





