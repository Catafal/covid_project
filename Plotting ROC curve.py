import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
%matplotlib inline

# Read the datafile "covid_train.csv"
df_train = pd.read_csv('covid_train.csv')

# Read the datafile "covid_test.csv"
df_test = pd.read_csv('covid_test.csv')

# Get the train predictors
X_train = df_train.drop('urgency', axis=1)
# Get the train response variable
y_train = df_train['urgency']

# Get the test predictors
X_test = df_test.drop('urgency', axis=1)

# Get the test response variable
y_test = df_test['urgency']

# kNN classification model with k = 7
knn = KNeighborsClassifier(n_neighbors = 7)

# Fit the above model on the train data
knn.fit(X_train,y_train)

# Predict probabilities for the positive class on the test data using the kNN model
y_pred_knn = knn.predict_proba(X_test)[:, 1]

# Logistic Regression model with max_iter as 10000, C as 0.1, and a random_state of 42
logreg = LogisticRegression(C=0.1, random_state=42, max_iter=10000)

# Fit the Logistic Regression model on the train data
logreg.fit(X_train, y_train)

# Predict probabilities for the positive class on the test data using the logistic regression model
y_pred_logreg = logreg.predict_proba(X_test)[:, 1]

###ROC Curve Review
###The Bayes threshold of a binary classifier is the value for which all predicted probabilities greater than or equal to that value are labeled as the positive class. For example, a classifier with a Bayes threshold of 0.6 will classify all observations with a predicted probability,
###p ≥0.6, as the positive class (1) and all observations with a predicted probability, p<0.6, as the negative class (0).
###The ROC curve shows us a model's false positive and true positive rates across different settings of the Bayes Threshold.
###We will compute the false positve rate (FPR) and true positive rate (TPR) for a range of thresholds and use these values to plot ROC curves for both the kNN and the logistic regression models.

def get_thresholds(y_pred_proba):
    # We only need to consider unique predicted probabilities
    unique_probas = np.unique(y_pred_proba)
    # Sort unique probabilities in descending order
    unique_probas_sorted = np.sort(unique_probas)[::-1]

    # We'll also add some additional thresholds to our set
    # This ensures our ROC curves reach the corners of the plot, (0,0) and (1,1)
    thresholds = np.insert(unique_probas_sorted, 0, 1.1)
    # Append 0 to the end of the thresholds
    thresholds = np.append(thresholds, 0)
    return thresholds

knn_thresholds = get_thresholds(y_pred_knn)

logreg_thresholds = get_thresholds(y_pred_logreg)

###FPR & TPR
###Now we can use the true y class label and the predicted probabilities to determine the the fpr and tpr on the test data for a specific threshold.

def get_fpr(y_true, y_pred_proba, threshold):
    # Predicted positive class labels based on the threshold
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # True positive, true negative, false positive, false negative counts
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    
    # False positive rate calculation
    fpr = fp / (fp + tn)
    
    return fpr


def get_tpr(y_true, y_pred_proba, threshold):
# Predicted positive class labels based on the threshold
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # True positive, true negative, false positive, false negative counts
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    
    # False positive rate calculation
    tpr = tp / (tp + fn)
    
    return tpr


###Use these functions to get the FPR and TPR for both models using each threshold for that model.

# FPR for the kNN at each of its thresholds
knn_fpr = [get_fpr(y_test, y_pred_knn, threshold) for threshold in knn_thresholds]

# TPR for the kNN at each of its thresholds
knn_tpr = [get_tpr(y_test, y_pred_knn, threshold) for threshold in knn_thresholds]

# TPR for the logistic model at each of its thresholds
logreg_tpr = [get_tpr(y_test, y_pred_logreg, threshold) for threshold in logreg_thresholds]

# FPR for the logistic model at each of its thresholds
logreg_fpr = [get_fpr(y_test, y_pred_logreg, threshold) for threshold in logreg_thresholds]

###Area Under the Curve
###The AUC gives us an idea as to how well our model does across all thresholds.
# Compute the ROC AUC score of the Logistic model
knn_auc = roc_auc_score(y_test, y_pred_knn)

# Compute the ROC AUC score of the kNN model
logreg_auc = roc_auc_score(y_test, y_pred_logreg)


# Area under curve - Logistic Regression & kNN
fig, ax = plt.subplots(figsize = (14,8))

# Plot KNN Regression ROC Curve
ax.plot(knn_fpr,
        knn_tpr,
        label=f'KNN (area = {knn_auc:.2f})',
        color='g',
        lw=3)

# Plot Logistic Regression ROC Curve
ax.plot(logreg_fpr,
        logreg_tpr,
        label=f'Logistic Regression (area = {logreg_auc:.2f})',
        color = 'purple',
        lw=3)

# Threshold annotations
label_kwargs = {}
label_kwargs['bbox'] = dict(
    boxstyle='round, pad=0.3', color='lightgray', alpha=0.6
)
eps = 0.02 # offset
for i in range(0, len(logreg_fpr),15):
    threshold = str(np.round(logreg_thresholds[i], 2))
    ax.annotate(threshold, (logreg_fpr[i], logreg_tpr[i]-eps), fontsize=12, color='purple', **label_kwargs)

for i in range(0, len(knn_fpr)-1):
    threshold = str(np.round(knn_thresholds[i], 2))
    ax.annotate(threshold, (knn_fpr[i], knn_tpr[i]+eps), fontsize=12, color='green', **label_kwargs)

# Plot diagonal line representing a random classifier
ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')

# Scenario 1 - Brazil
ax.fill_between([0,0.5],[0.5,0], color = 'red', alpha = 0.4, label='Scenario 1 - Brazil');

# Scenario 2 - Germany
ax.axhspan(0.8, 0.9, facecolor='y', alpha=0.4, label = 'Scenario 2 - Germany');

# Scenario 3 - India
ax.fill_between([0,1],[1,0],[0.5,-0.5], alpha = 0.4, color = 'blue', label = 'Scenario 3 - India');

ax.set_xlim([0.0, 1.0]);
ax.set_ylim([0.0, 1.05]);
ax.set_xlabel('False Positive Rate', fontsize=20)
ax.set_ylabel('True Positive Rate', fontsize=20)
ax.set_title('Receiver Operating Characteristic', fontsize=20)
ax.legend(loc="lower right", fontsize=15)
plt.show()

