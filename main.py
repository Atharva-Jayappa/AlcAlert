import numpy as np
import pandas as pd
import os
import random
from tqdm import tqdm
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly import tools
from scipy.stats import mannwhitneyu


seed = 123
random.seed = seed

# C:\Users\jatha\Desktop\Sem5\EDI\SMNI_CMI_TRAIN\Train
# data import
filenames_list = os.listdir(
    'C:/Users/jatha/Desktop/Sem5/EDI/SMNI_CMI_TRAIN/Train')  ## list of file names in the directory
EEG_data = pd.DataFrame({})  ## create an empty df that will hold data from each file
print(len(filenames_list))

for file_name in tqdm(filenames_list):
    temp_df = pd.read_csv('C:/Users/jatha/Desktop/Sem5/EDI/SMNI_CMI_TRAIN/Train/' + file_name)

    # Concatenate the current DataFrame with EEG_data
    EEG_data = pd.concat([EEG_data, temp_df], ignore_index=True)

# Reset the index of the final concatenated DataFrame
EEG_data.reset_index(drop=True, inplace=True)
EEG_data = EEG_data.drop(['Unnamed: 0'], axis=1)  ## remove the unused column
EEG_data.loc[EEG_data[
                 'matching condition'] == 'S2 nomatch,', 'matching condition'] = 'S2 nomatch'  ## remove comma sign from stimulus name
# example try for create train set
temp_df = pd.read_csv('C:/Users/jatha/Desktop/Sem5/EDI/SMNI_CMI_TRAIN/Train/Data1.csv')
print(temp_df['subject identifier'][0])
print(temp_df["matching condition"][0])
print(np.array(temp_df["sensor value"]).reshape([64, 256]))

# S1: S1 obj - a single object shown;
s1 = 0
# S12: S2 nomatch - object 2 shown in a non matching condition (S1 differed from S2)
s12 = 0
# S21: S2 match - object 2 shown in a matching condition (S1 was identical to S2),
s21 = 0
s1_X_train = np.zeros((160, 256, 64))
s1_y_train = np.zeros(160)
s21_X_train = np.zeros((159, 256, 64))
s21_y_train = np.zeros(159)
s12_X_train = np.zeros((149, 256, 64))
s12_y_train = np.zeros(149)

classifier = {'a': 1, 'c': 0}

filenames_list = os.listdir(
    'C:/Users/jatha/Desktop/Sem5/EDI/SMNI_CMI_TRAIN/Train')  ## list of file names in the directory
EEG_data = pd.DataFrame({})
print(len(filenames_list))
for file_name in tqdm(filenames_list):
    temp_df = pd.read_csv(
        'C:/Users/jatha/Desktop/Sem5/EDI/SMNI_CMI_TRAIN/Train/' + file_name)  ## read from the file to df
    if temp_df["matching condition"][0] == "S1 obj":
        s1_X_train[s1] = np.transpose(np.array(temp_df["sensor value"]).reshape([64, 256]))
        s1_y_train[s1] = classifier[temp_df['subject identifier'][0]]
        s1 += 1
    if temp_df["matching condition"][0] == "S2 match":
        s21_X_train[s21] = np.transpose(np.array(temp_df["sensor value"]).reshape([64, 256]))
        s21_y_train[s21] = classifier[temp_df['subject identifier'][0]]
        s21 += 1
    if temp_df["matching condition"][0] == "S2 nomatch,":
        s12_X_train[s12] = np.transpose(np.array(temp_df["sensor value"]).reshape([64, 256]))
        s12_y_train[s12] = classifier[temp_df['subject identifier'][0]]
        s12 += 1
print(s1)
print(s21)
print(s12)

# t1: S1 obj - a single object shown;
t1 = 0
# t12: S2 nomatch - object 2 shown in a non matching condition (S1 differed from S2)
t12 = 0
# t21: S2 match - object 2 shown in a matching condition (S1 was identical to S2),
t21 = 0
t1_X_test = np.zeros((160, 256, 64))
t1_y_test = np.zeros(160)
t21_X_test = np.zeros((160, 256, 64))
t21_y_test = np.zeros(160)
t12_X_test = np.zeros((160, 256, 64))
t12_y_test = np.zeros(160)

classifier = {'a': 1, 'c': 0}

filenames_list = os.listdir(
    'C:\\Users\\jatha\Desktop\\Sem5\\EDI\\SMNI_CMI_TEST')  ## list of file names in the directory
EEG_data = pd.DataFrame({})  ## create an empty df that will hold data from each file
print(len(filenames_list))
for file_name in tqdm(filenames_list):
    if file_name == "Test":
        pass
    else:
        temp_df = pd.read_csv(
            'C:\\Users\\jatha\\Desktop\\Sem5\\EDI\\SMNI_CMI_TEST/' + file_name)  ## read from the file to df
        if temp_df["matching condition"][0] == "S1 obj":
            t1_X_test[t1] = np.transpose(np.array(temp_df["sensor value"]).reshape([64, 256]))
            t1_y_test[t1] = classifier[temp_df['subject identifier'][0]]
            t1 += 1
        if temp_df["matching condition"][0] == "S2 match":
            t21_X_test[t21] = np.transpose(np.array(temp_df["sensor value"]).reshape([64, 256]))
            t21_y_test[t21] = classifier[temp_df['subject identifier'][0]]
            t21 += 1
        if temp_df["matching condition"][0] == "S2 nomatch,":
            t12_X_test[t12] = np.transpose(np.array(temp_df["sensor value"]).reshape([64, 256]))
            t12_y_test[t12] = classifier[temp_df['subject identifier'][0]]
            t12 += 1
print(t1)
print(t21)
print(t12)

from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tqdm import tqdm

for i in tqdm(range(3, 20)):
    knn = KNeighborsTimeSeriesClassifier(n_neighbors=i, n_jobs=12)
    knn.fit(s1_X_train, s1_y_train)
    print(knn.score(t1_X_test, t1_y_test))
from tslearn.svm import TimeSeriesSVC

clf = TimeSeriesSVC(C=1.0, kernel="gak")
clf.fit(s1_X_train, s1_y_train)
print(clf.score(t1_X_test, t1_y_test))

mean_train = np.mean(s1_X_train, axis=1)
std_train = np.std(s1_X_train, axis=1)

# Calculate the mean and standard deviation for each sensor/channel in test data
mean_test = np.mean(t1_X_test, axis=1)
std_test = np.std(t1_X_test, axis=1)

# Combine the calculated statistics into feature representations for both training and test data
s1_sig_train = np.column_stack((mean_train, std_train))
s1_sig_test = np.column_stack((mean_test, std_test))

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
clf.fit(s1_sig_train, s1_y_train)
y_pred_proba = clf.predict_proba(s1_sig_test)[:, 1]
y_pred = clf.predict(s1_sig_test)

from plot_metric.functions import BinaryClassification
import matplotlib.pyplot as plt
# Visualisation with plot_metric
from sklearn.metrics import accuracy_score

bc = BinaryClassification(t1_y_test, y_pred_proba, labels=["Class 0", "Class 1"])
# Figures
plt.figure(figsize=(5, 5))
bc.plot_roc_curve()
plt.title("Receiver Operating Characteristic Using Random Forest")
plt.show()
score = accuracy_score(t1_y_test, y_pred)
print("score is " + str(score))

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(s1_sig_train, s1_y_train)
y_pred_proba = clf.predict_proba(s1_sig_test)[:, 1]
y_pred = clf.predict(s1_sig_test)

from plot_metric.functions import BinaryClassification
import matplotlib.pyplot as plt
# Visualisation with plot_metric
from sklearn.metrics import accuracy_score

bc = BinaryClassification(t1_y_test, y_pred_proba, labels=["Class 0", "Class 1"])
# Figures
plt.figure(figsize=(5, 5))
bc.plot_roc_curve()
plt.title("Receiver Operating Characteristic Using Logistic Regression")
plt.show()
score = accuracy_score(t1_y_test, y_pred)
print("score is " + str(score))

from sklearn.svm import SVC

clf = SVC(probability=True)
clf.fit(s1_sig_train, s1_y_train)
y_pred_proba = clf.predict_proba(s1_sig_test)[:, 1]
y_pred = clf.predict(s1_sig_test)

from plot_metric.functions import BinaryClassification
import matplotlib.pyplot as plt
# Visualisation with plot_metric
from sklearn.metrics import accuracy_score

bc = BinaryClassification(t1_y_test, y_pred_proba, labels=["Class 0", "Class 1"])
# Figures
plt.figure(figsize=(5, 5))
bc.plot_roc_curve()
plt.title("Receiver Operating Characteristic Using SVC")
plt.show()
score = accuracy_score(t1_y_test, y_pred)
print("score is " + str(score))
from sklearn.neighbors import KNeighborsClassifier

for i in tqdm(range(3, 20)):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(s1_sig_train, s1_y_train)
    y_pred = knn.predict(s1_sig_test)
    print(accuracy_score(t1_y_test, y_pred))
from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier()
clf.fit(s1_sig_train, s1_y_train)
y_pred_proba = clf.predict_proba(s1_sig_test)[:, 1]
y_pred = clf.predict(s1_sig_test)

from plot_metric.functions import BinaryClassification
import matplotlib.pyplot as plt
# Visualisation with plot_metric
from sklearn.metrics import accuracy_score

bc = BinaryClassification(t1_y_test, y_pred_proba, labels=["Class 0", "Class 1"])
# Figures
plt.figure(figsize=(5, 5))
bc.plot_roc_curve()
plt.title("Receiver Operating Characteristic Using AdaBoostClassifier")
plt.show()
score = accuracy_score(t1_y_test, y_pred)
print("score is " + str(score))

from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tqdm import tqdm

for i in tqdm(range(3, 20)):
    knn = KNeighborsTimeSeriesClassifier(n_neighbors=i, n_jobs=12)
    knn.fit(s12_X_train, s12_y_train)
    print(knn.score(t12_X_test, t12_y_test))
from tslearn.svm import TimeSeriesSVC
import datetime

start = datetime.datetime.now()
clf = TimeSeriesSVC(C=1.0, kernel="gak")
clf.fit(s12_X_train, s12_y_train)
score = clf.score(t12_X_test, t12_y_test)
end = datetime.datetime.now()
print(score)
print(end - start)

mean_train = np.mean(s12_X_train, axis=1)
std_train = np.std(s12_X_train, axis=1)

# Calculate the mean and standard deviation for each sensor/channel in test data
mean_test = np.mean(t12_X_test, axis=1)
std_test = np.std(t12_X_test, axis=1)

# Combine the calculated statistics into feature representations for both training and test data
s12_sig_train = np.column_stack((mean_train, std_train))
s12_sig_test = np.column_stack((mean_test, std_test))

from sklearn.ensemble import RandomForestClassifier

start = datetime.datetime.now()
clf = RandomForestClassifier()
clf.fit(s12_sig_train, s12_y_train)
y_pred_proba = clf.predict_proba(s12_sig_test)[:, 1]
y_pred = clf.predict(s12_sig_test)

from plot_metric.functions import BinaryClassification
import matplotlib.pyplot as plt
# Visualisation with plot_metric
from sklearn.metrics import accuracy_score

bc = BinaryClassification(t12_y_test, y_pred_proba, labels=["Class 0", "Class 1"])
# Figures
plt.figure(figsize=(5, 5))
bc.plot_roc_curve()
plt.title("Receiver Operating Characteristic Using Random Forest")
plt.show()
score = accuracy_score(t12_y_test, y_pred)
end = datetime.datetime.now()
print("score is " + str(score))
print(end - start)

from sklearn.linear_model import LogisticRegression

start = datetime.datetime.now()
clf = LogisticRegression()
clf.fit(s12_sig_train, s12_y_train)
y_pred_proba = clf.predict_proba(s12_sig_test)[:, 1]
y_pred = clf.predict(s12_sig_test)

from plot_metric.functions import BinaryClassification
import matplotlib.pyplot as plt
# Visualisation with plot_metric
from sklearn.metrics import accuracy_score

bc = BinaryClassification(t12_y_test, y_pred_proba, labels=["Class 0", "Class 1"])
# Figures
plt.figure(figsize=(5, 5))
bc.plot_roc_curve()
plt.title("Receiver Operating Characteristic Using LogisticRegression")
plt.show()
score = accuracy_score(t12_y_test, y_pred)
end = datetime.datetime.now()
print("score is " + str(score))
print(end - start)
from sklearn.svm import SVC

start = datetime.datetime.now()
clf = SVC(probability=True)
clf.fit(s12_sig_train, s12_y_train)
y_pred_proba = clf.predict_proba(s12_sig_test)[:, 1]
y_pred = clf.predict(s12_sig_test)

from plot_metric.functions import BinaryClassification
import matplotlib.pyplot as plt
# Visualisation with plot_metric
from sklearn.metrics import accuracy_score

bc = BinaryClassification(t12_y_test, y_pred_proba, labels=["Class 0", "Class 1"])
# Figures
plt.figure(figsize=(5, 5))
bc.plot_roc_curve()
plt.title("Receiver Operating Characteristic Using SVC")
plt.show()
score = accuracy_score(t12_y_test, y_pred)
end = datetime.datetime.now()
print("score is " + str(score))
print(end - start)
from sklearn.neighbors import KNeighborsClassifier

for i in tqdm(range(3, 20)):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(s12_sig_train, s12_y_train)
    y_pred = knn.predict(s12_sig_test)
    print(accuracy_score(t12_y_test, y_pred))
from sklearn.ensemble import AdaBoostClassifier

start = datetime.datetime.now()
clf = AdaBoostClassifier()
clf.fit(s12_sig_train, s12_y_train)
y_pred_proba = clf.predict_proba(s12_sig_test)[:, 1]
y_pred = clf.predict(s12_sig_test)

from plot_metric.functions import BinaryClassification
import matplotlib.pyplot as plt
# Visualisation with plot_metric
from sklearn.metrics import accuracy_score

bc = BinaryClassification(t12_y_test, y_pred_proba, labels=["Class 0", "Class 1"])
# Figures
plt.figure(figsize=(5, 5))
bc.plot_roc_curve()
plt.title("Receiver Operating Characteristic Using AdaBoostClassifier")
plt.show()
score = accuracy_score(t12_y_test, y_pred)
end = datetime.datetime.now()
print("score is " + str(score))
print(end - start)
