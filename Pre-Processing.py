import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
%matplotlib inline

from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split

# Read the datafile "covid.csv"
df = pd.read_csv('covid.csv')

# Check if there are any missing or Null values
pd.isna(df)

# Find the number of rows with missing values
# Count the number of missing values in each row
missing_rows = df.isnull().sum(axis=1)

# Count the total number of rows with missing values
num_null = (missing_rows > 0).sum()
print("Number of rows with null values:", num_null)

# kNN impute the missing data
# Use a k value of 5
columns_with_missing_values = ['age','sex','cough','fever','chills','sore_throat','headache','fatigue']

imputed = KNNImputer(n_neighbors=5).fit_transform(df[columns_with_missing_values])

# Replace the original dataframe with the imputed data, continue to use df for the dataframe
imputed_df = pd.DataFrame(imputed, columns = columns_with_missing_values)

df[columns_with_missing_values] = imputed_df

df.head()

## EDA (understanding of the data with some questions)
# Plot an appropriate graph to answer the following question
# Your code here
plt.hist(df['age'], bins = 50)
plt.xlabel('Age')
plt.ylabel('Urgency')
plt.title('Knn plot with age')
plt.show()

##⏸ Which age group has the most urgent need for a hospital bed? -> 40 - 50   
columns_to_plot = ['cough', 'fever', 'sore_throat', 'fatigue']
fig, axs = plt.subplots(1, len(columns_to_plot), figsize=(12, 4))

for i, column in enumerate(columns_to_plot):
    axs[i].hist(df[column])
    axs[i].set_title(column)

plt.tight_layout()
plt.show()

## Among the following symptoms, which is the most common one for patients with urgent need of hospitalization? -> Fever
# Filter the dataset for patients with urgent need of hospitalization
urgent_patients = df[df['Urgency'] == 1.0]
print(urgent_patients)
# Filter the dataset for patients with no urgency
non_urgent_patients = df[df['Urgency'] == 0.0]
print(non_urgent_patients)

# Calculate the frequency of cough for each group
cvalues_urgent = urgent_patients['cough'].values
urgent_cough_frequency = np.count_nonzero(cvalues_urgent == 1)
cvalues_non_urgent = non_urgent_patients['cough'].values
non_urgent_cough_frequency = np.count_nonzero(cvalues_non_urgent == 0)

# Create a bar plot
fig, ax = plt.subplots()
x = ['With Urgency', 'No Urgency']
y = [urgent_cough_frequency, non_urgent_cough_frequency]
ax.bar(x, y)

# Set the plot title and axis labels
ax.set_title('Frequency of Cough by Urgency')
ax.set_xlabel('Urgency')
ax.set_ylabel('Cough')

# Display the plot
plt.show()

## As compared to patients with urgent need of hospitalization, patients with no urgency have cough as a more common symptom? -> True

# Split the data into train and test sets with 70% for training
df_train, df_test = train_test_split(df, test_size = 0.3, random_state =60)

# Save the train data into a csv called "covid_train.csv"
# Remember to not include the default indices
df_train.to_csv('covid_train.csv', index=False)

# Save the test data into a csv called "covid_test.csv"
# Remember to not include the default indices
df_train.to_csv('covid_test.csv', index=False)
