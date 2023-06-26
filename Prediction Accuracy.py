import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
%matplotlib inline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Read the datafile "covid_train.csv"
df_train = pd.read_csv('covid_train.csv')

# Read the datafile "covid_test.csv"
df_test = pd.read_csv('covid_test.csv')

# Get the train predictors
X_train = df_train.drop('Urgency', axis=1)

# Get the train response variable
y_train = df_train[['Urgency']]

# Get the test predictors
X_test = df_test.drop('Urgency',axis=1)

# Get the test response variable
y_test = df_test[['Urgency']]

# Classification model
model = KNeighborsClassifier(n_neighbors=5)

# Fit the model on the train data
model.fit(X_train,y_train)

# Predict and compute the accuracy on the test data
y_pred = model.predict(X_test)

model_accuracy = accuracy_score(y_test,y_pred)
print(f"Model Accuracy is {model_accuracy}")

