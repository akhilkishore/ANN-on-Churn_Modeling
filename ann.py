import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

### Part 1: Data Pre-processing ###
# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#encode categorical datas
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])

labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])

onehotencoder = OneHotEncoder(categorical_features = [1]) #changes only on index 1
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] #remove one dummy variable out of 3 dummy vairables we created

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling a.k.a Normalization 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


### Part 2 : Create ANN ###
































































































#academic_project for noobies  developed by @akhilkisore ,Thanks to www.