import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

diabetes_data = pd.read_csv('diabetes.csv')
diabetes_data.head()

diabetes_data.info()

diabetes_data.isnull().sum()

diabetes_data.describe()

diabetes_data.shape

diabetes_data['Outcome'].value_counts()

x= diabetes_data.drop(columns=['Outcome'],axis=1)
y=diabetes_data['Outcome']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)

model = LogisticRegression(random_state=42)
model.fit(x_train,y_train)

x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(y_train,x_train_prediction)

print('Accuracy score of training data:',training_data_accuracy)

x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(y_test,x_test_prediction)

print('Accuracy score of training data:',test_data_accuracy)

input_data = [
    119.992, 157.302, 74.997, 0.00784, 0.04374, 0.01164, 21.033,
    10
]

input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if(prediction[0]==0):
  print('Non Diabetic Person')
else:
  print('Diabetic Person')


  import pickle
  filename = 'diabetes_model.sav'
pickle.dump(model,open(filename,'wb'))

loaded_model = pickle.load(open('diabetes_model.sav','rb'))