import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
%matplotlib inline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
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
X_test = df_test.drop('Urgency', axis=1)

# Get the test response variable
y_test = df_test[['Urgency']]

# Define a kNN classification model with k = 7
knn_model = KNeighborsClassifier(n_neighbors = 7)

# Fit the above model on the train data
knn_model.fit(X_train, y_train)
# Logistic Regression model with max_iter as 10000 and C as 0.1 (leave all other parameters at default values)
log_model = LogisticRegression(C=0.1, max_iter=10000)

# Fit the Logistic Regression model on the train data
log_model.fit(X_train, y_train)

##Now it is time to evaluate the models.

metric_scores = {'Accuracy': [], 'Recall': [], 'Specificity': [], 'Precision': [], 'F1-score': []}

y_pred_knn = knn_model.predict(X_test)
y_pred_log = log_model.predict(X_test)
tn_knn, fp_knn, fn_knn, tp_knn = confusion_matrix(y_test, y_pred_knn).ravel()
tn_log, fp_log, fn_log, tp_log = confusion_matrix(y_test, y_pred_log).ravel()

metric_scores['Accuracy'] = [accuracy_score(y_test, y_pred_knn), accuracy_score(y_test, y_pred_log)]
metric_scores['Recall'] = [(tp_knn/(tp_knn+fn_knn)), (tp_log/(tp_log+fn_log))]
metric_scores['Specificity'] = [(tn_knn/(tn_knn+fp_knn)), (tn_log/(tn_log+fp_log))]
metric_scores['Precision'] = [(tp_knn/(tp_knn+fp_knn)), (tp_log/(tp_log+fp_log))]
metric_scores['F1-score'] = [(tp_knn/(tp_knn+1/2*(fp_knn+fn_knn))), (tp_log/(tp_log+1/2*(fp_log+fn_log)))]

# Display your results
print(metric_scores)
