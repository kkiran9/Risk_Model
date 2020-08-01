# import libraries
import pandas as pd
import numpy as np
# import data
df = pd.read_csv("..\creditcard.csv")
# view the column names
df.columns

# number of fraud and non-fraud observations in the dataset
frauds = len(df[df.Class == 1])
nonfrauds = len(df[df.Class == 0])
print("Frauds", frauds); print("Non-frauds", nonfrauds)
## scaling the "Amount" and "Time" columns similar to the others variables
from sklearn.preprocessing import RobustScaler
rob_scaler = RobustScaler()
df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))
# now drop the original columns
df.drop(['Time','Amount'], axis=1, inplace=True)
# define X and y variables
X = df.loc[:, df.columns != 'Class']
y = df.loc[:, df.columns == 'Class']

# number of fraud cases
frauds = len(df[df.Class == 1])
# selecting the indices of the non-fraud classes
fraud_indices = df[df.Class == 1].index
nonfraud_indices = df[df.Class == 0].index
# From all non-fraud observations, randomly select observations equal to number of fraud observations
random_nonfraud_indices = np.random.choice(nonfraud_indices, frauds, replace = False)
random_nonfraud_indices = np.array(random_nonfraud_indices)
# Appending the 2 indices
under_sample_indices = np.concatenate([fraud_indices,random_nonfraud_indices])
# Under sample dataset
under_sample_data = df.iloc[under_sample_indices,:]
# Now split X, y variables from the under sample data
X_undersample = under_sample_data.loc[:, under_sample_data.columns != 'Class']
y_undersample = under_sample_data.loc[:, under_sample_data.columns == 'Class']

## split data into training and testing set
from sklearn.model_selection import train_test_split
# # The complete dataset
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)
# Split dataset
X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(X_undersample,y_undersample                                                                                                                                                                                                                                                                                             ,random_state = 0)
## modeling with logistic regression
#import model
from sklearn.linear_model import LogisticRegression
# instantiate model
model = LogisticRegression()
# fit 
model.fit(X_train_undersample, y_train_undersample)
# predict
y_pred = model.predict(X_test_undersample)

# import classification report and confusion matrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
classification_report = classification_report(y_test_undersample, y_pred)
confusion_matrix = confusion_matrix(y_test_undersample, y_pred)
print("CLASSIFICATION REPORT")
print(classification_report)
print("CONFUSION MATRIX") 
print(confusion_matrix)