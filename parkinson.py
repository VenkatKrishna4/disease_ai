import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import svm
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


parkinsons_data = pd.read_csv('Parkinsson disease.csv')

print(parkinsons_data.head())

parkinsons_data.shape

parkinsons_data.info()

parkinsons_data.isnull().sum()

parkinsons_data.describe()


#distribution of target variable
parkinsons_data['status'].value_counts()

parkinsons_data.groupby('status').mean(numeric_only=True)

x = parkinsons_data.drop(columns=['name','status'], axis=1)
y = parkinsons_data['status']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=2)

model = svm.SVC(kernel='linear')
model.fit(x_train,y_train)

x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(y_train,x_train_prediction)

print('Accuracy score of training data:',training_data_accuracy)

x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(y_test,x_test_prediction)

print('Accuracy score of training data:',test_data_accuracy)

input_data = [
    119.992, 157.302, 74.997, 0.00784, 0.04374, 0.01164, 21.033,
    0.0121, 0.0193, 0.0217, 0.0235, 0.0189, 0.0171, 0.0062, 0.022, 
    0.015, 0.014, 0.021, 0.018, 0.0145, 0.016, 0.013 
]

input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if(prediction[0]==0):
  print('The person does not have Parkinson’s Disease')
else:
  print('The person has Parkinson’s Disease')


import pickle

filename ='parkinson_model.sav'
pickle.dump(model,open(filename,'wb'))

loaded_model = pickle.load(open('parkinson_model.sav','rb'))