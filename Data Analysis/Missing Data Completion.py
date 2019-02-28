# -*- coding: utf-8 -*-

#definition of libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer

missingdata = pd.read_csv('data.csv')


imputer = Imputer(missing_values='NaN',strategy='mean', axis=0)

#non-numeric columns not included
data = missingdata.iloc[:,1:4].values 

imputer = imputer.fit(data[:,1:4])
#Write an average of the data instead of Nan
data[:,1:4] = imputer.transform(data[:,1:4])


print(data)
