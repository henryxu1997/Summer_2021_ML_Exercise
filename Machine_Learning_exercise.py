import csv
import pandas as pd
import re
import seaborn as sn
import matplotlib.pyplot as plt
import os
import numpy as np
import glob
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Using a dictionary for label encoding for benign and malicious attacks
encoding = {'benign': {'-': 0}, 'malicious': {'c&c': 1, 'c&c-filedownload': 1, 'c&c-heartbeat': 2, 
'c&c-heartbeat-attack': 2, 'c&c-heartbeat-filedownload': 2, 'c&c-mirai': 3, 'c&c-torii': 4, 'ddos': 5, 
'filedownload': 6, 'okiru': 7, 'okiru-attack': 7, 'partofahorizontalportscan': 8, 'partofahorizontalportscan-attack': 8, 
'c&c-partofahorizontalportscan': 8, 'attack': 9}}

# File path
path = r"C:\Users\henry\Desktop\Summer_Research\iot_23_datasets_small.tar\opt\Malware-Project\BigDataset\IoTScenarios"


# Traverse all files in all subfolders of the current folder
all_files = []
for root, dirs, files in os.walk(path):
	for file in files:
		all_files.append(os.path.join(root, file))

# For all files found, load the data in with specified column names
# Additionally, include deliminiter to include all whitespace, then skip first 8 rows and last row of each dataset
# Lastly, only store the first 10,000 rows of data as computation resource is limited
print("This is all files {}".format(all_files))
all_dataFrames = []
for filename in all_files:
	print("Current file name: {}".format(filename))
	csv_data = pd.read_csv(filename, names = ["ts", "uid",	"id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p", "proto", 
		"service", "duration", "orig_bytes", "resp_bytes", "conn_stat", "local_orig", "local_resp", "missed_bytes","history", 
		"orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes", "tunnel_parents", "label", "detailed-label"], 
		delim_whitespace=True, skiprows = [i for i in range(8)], skipfooter = 1)

	current_df = pd.DataFrame(csv_data, columns = ["ts", "uid",	"id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p", "proto", 
		"service", "duration", "orig_bytes", "resp_bytes", "conn_stat", "local_orig", "local_resp", "missed_bytes","history", 
		"orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes", "tunnel_parents", "label", "detailed-label"]).head(10000)
	all_dataFrames.append(current_df)
df = pd.concat(all_dataFrames)

# For non-numerical columns, encode them with numerical values so they can be used
df['id.orig_h'] = df['id.orig_h'].astype('category').cat.codes
df['uid'] = df['uid'].astype('category').cat.codes
df['id.resp_h'] = df['service'].astype('category').cat.codes
df['proto'] = df['proto'].astype('category').cat.codes
df['service'] = df['service'].astype('category').cat.codes
df['duration'] = df['duration'].astype('category').cat.codes
df['orig_bytes'] = df['orig_bytes'].astype('category').cat.codes
df['resp_bytes'] = df['resp_bytes'].astype('category').cat.codes
df['conn_stat'] = df['conn_stat'].astype('category').cat.codes
df['local_orig'] = df['local_orig'].astype('category').cat.codes
df['local_resp'] = df['local_resp'].astype('category').cat.codes
df['history'] = df['history'].astype('category').cat.codes
df['tunnel_parents'] = df['tunnel_parents'].astype('category').cat.codes

# Standardize the strings to be all lowered case
df['label'] = df['label'].str.lower()
df['detailed-label'] = df['detailed-label'].str.lower()

# Encoding function for each type of attack. This function will access a nested diciontary using the label and detailed-label as keys
def get_encoding(t, s):
	v = encoding[s][t]
	return v

# Apply the enocding to the values for label encoding according to the table in paper
df['New_Label'] = df.apply(lambda x: get_encoding(x['detailed-label'], x['label']), axis = 1)

# Labels are the values we want to predict
original_labels = np.array(df['New_Label'])

# Drop irrelevant columns that do not have enough correlation to the labels
df = df.drop('New_Label', axis = 1)
df = df.drop('detailed-label', axis = 1)
df = df.drop('label', axis = 1)
df = df.drop('ts', axis = 1)
df = df.drop('uid', axis = 1)
df = df.drop('id.orig_h', axis = 1)
df = df.drop('local_orig', axis = 1)
df = df.drop('local_resp', axis = 1)
df = df.drop('missed_bytes', axis = 1)
df = df.drop('tunnel_parents', axis = 1)


# Saving feature names
original_feature_list = list(df.columns)

# Convert the data scheme to numpy array
original_features = np.array(df)

# Split the data into training and testing sets with 80-20 split specified on paper
original_train_features, original_test_features, original_train_labels, original_test_labels = train_test_split(
	original_features, original_labels, test_size = 0.2, random_state = 42)

print("This is original train features: {}".format(original_train_features))
print("This is original_test_features: {}".format(original_test_features))
print("This is original_train_labels: {}".format(original_train_labels))
print("This is original_test_labels: {}".format(original_test_labels))


# Import the model we are using
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier

# Import the metrics calculation libraries 
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# XGBoost
# Instantiate model 
# Train the model on training data
# Make a prediction on the test data
algorithm_name = "XGBoost"
xgb_model = XGBClassifier(objective="multi:softprob", random_state = 42)
xgb_model.fit(original_train_features, original_train_labels)
predictions = xgb_model.predict(original_test_features)

# Decision tree
algorithm_name = "Decision Tree"
dTree = DecisionTreeClassifier()
dTree.fit(original_train_features, original_train_labels)
predictions = dTree.predict(original_test_features)

# Random forest
algorithm_name = "Random Forest"
rf = RandomForestClassifier(n_estimators= 1, random_state=42)
rf.fit(original_train_features, original_train_labels);
predictions = rf.predict(original_test_features)

# Naive Bayes classifier
algorithm_name = "Naive_Bayes"
gnb = GaussianNB()
gnb.fit(original_train_features, original_train_labels)
predictions = gnb.predict(original_test_features)

# SVM
# SVM default decision function shape is one versus all/rest
# Kernel selection: test and see which one performs the best
# Cannot experiment on the kernel as SVM takes way too long to run even with small dataset
algorithm_name = "SVM"
svm = svm.SVC(kernel='linear')
svm.fit(original_train_features, original_train_labels)
predictions = svm.predict(original_test_features)

# Artificial Neural Network with MLP
# The number of nodes on each layer isnt specified in paper. Additionally, activation function isnt specified.
algorithm_name = "MLP"
mlp = MLPClassifier(hidden_layer_sizes = (256, 128, 64, 32), activation = "relu", random_state = 1)
mlp.fit(original_train_features, original_train_labels)
predictions = mlp.predict(original_test_features)

# AdaBoosts
algorithm_name = "AdaBoost"
abc = AdaBoostClassifier(n_estimators = 1000, learning_rate=1)
abc.fit(original_train_features, original_train_labels)
predictions = abc.predict(original_test_features)

# Print out useful information about the results, algorithm used, and amount of data used 
print("This is prediction: {}".format(predictions))
print("This is the {} algorithm".format(algorithm_name))
print("Number of rows of data used: {}".format(df.shape[0]))

# Analysis on the algorithms with metrics
# Adding labels=np.unique(predictions) to ignore non-predicted labels
weighted_precision = precision_score(original_test_labels, predictions, average='weighted', labels=np.unique(predictions))
print("This is the weighted precision score: {}".format(weighted_precision))

macro_precision = precision_score(original_test_labels, predictions, average='macro', labels=np.unique(predictions))
print("This is the macro precision score: {}".format(macro_precision))

weighted_recall = recall_score(original_test_labels, predictions, average='weighted')
print("This is the weighted recall score: {}".format(weighted_recall))

macro_recall = recall_score(original_test_labels, predictions, average='macro')
print("This is the macro recall score: {}".format(macro_recall))

weighted_f1 = f1_score(original_test_labels, predictions, average='weighted')
print("This is the weighted f1 score: {}".format(weighted_f1))

macro_f1 = f1_score(original_test_labels, predictions, average='macro')
print("This is the macro f1 score: {}".format(macro_f1))

accuracy = accuracy_score(original_test_labels, predictions)
print("This is the accuracy score: {}".format(accuracy))

# # Plot correlation matrix for each set of data. df here needs to be modified to individual dataframe instead of the overall one
corrMatrix = df.corr()
sn.heatmap(corrMatrix, cmap='coolwarm', vmin = -1, vmax = 1)
plt.show()
