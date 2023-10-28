import numpy as np
import pandas as pd
import os
import random
from tqdm import tqdm
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import datetime

seed = 123
random.seed = seed

# Data import
train_dir = 'C:/Users/jatha/Desktop/Sem5/EDI/SMNI_CMI_TRAIN/Train'
test_dir = 'C:/Users/jatha/Desktop/Sem5/EDI/SMNI_CMI_TEST'
classifier = {'a': 1, 'c': 0}


# Function to process data and create datasets
def process_data(data_dir, data_size):
    X = np.zeros((data_size, 256, 64))
    y = np.zeros(data_size)
    count = 0

    filenames_list = os.listdir(data_dir)
    for file_name in tqdm(filenames_list):
        temp_df = pd.read_csv(os.path.join(data_dir, file_name))

        if temp_df["matching condition"][0] == "S1 obj" or temp_df["matching condition"][0] == "S2 match":
            X[count] = np.transpose(np.array(temp_df["sensor value"]).reshape([64, 256]))
            y[count] = classifier[temp_df['subject identifier'][0]]
            count += 1

        if count >= data_size:
            break

    return X, y


# Create larger training datasets
train_size = 300
s1_X_train, s1_y_train = process_data(train_dir, train_size)
s12_X_train, s12_y_train = process_data(train_dir, train_size)
s21_X_train, s21_y_train = process_data(train_dir, train_size)

# Create test datasets (assuming the same size as training for simplicity)
test_size = train_size
t1_X_test, t1_y_test = process_data(test_dir, test_size)
t12_X_test, t12_y_test = process_data(test_dir, test_size)
t21_X_test, t21_y_test = process_data(test_dir, test_size)


# Train and evaluate KNN
print("Results using KNN:")
clf_knn = KNeighborsTimeSeriesClassifier(n_neighbors=5, n_jobs=12)
clf_knn.fit(np.concatenate((s1_X_train, s12_X_train, s21_X_train), axis=0),
            np.concatenate((s1_y_train, s12_y_train, s21_y_train), axis=0))

y_pred_proba_knn = clf_knn.predict_proba(np.concatenate((t1_X_test, t12_X_test, t21_X_test), axis=0))[:, 1]
y_pred_knn = clf_knn.predict(np.concatenate((t1_X_test, t12_X_test, t21_X_test), axis=0))

# Calculate ROC curve and AUC for KNN
fpr_knn, tpr_knn, _ = roc_curve(np.concatenate((t1_y_test, t12_y_test, t21_y_test), axis=0), y_pred_proba_knn)
roc_auc_knn = auc(fpr_knn, tpr_knn)

# Plot ROC curve for KNN
plt.figure(figsize=(5, 5))
plt.plot(fpr_knn, tpr_knn, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_knn)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("Receiver Operating Characteristic Using KNN")
plt.legend(loc="lower right")
plt.show()

# Calculate accuracy for KNN
score_knn = accuracy_score(np.concatenate((t1_y_test, t12_y_test, t21_y_test), axis=0), y_pred_knn)
print(f"Accuracy score using KNN: {score_knn}")

# Train and evaluate RandomForest
print("Results using RandomForest:")
clf_rf = RandomForestClassifier()
clf_rf.fit(np.concatenate((s1_X_train, s12_X_train, s21_X_train), axis=0),
           np.concatenate((s1_y_train, s12_y_train, s21_y_train), axis=0))

y_pred_proba_rf = clf_rf.predict_proba(np.concatenate((t1_X_test, t12_X_test, t21_X_test), axis=0))[:, 1]
y_pred_rf = clf_rf.predict(np.concatenate((t1_X_test, t12_X_test, t21_X_test), axis=0))

# Calculate ROC curve and AUC for RandomForest
fpr_rf, tpr_rf, _ = roc_curve(np.concatenate((t1_y_test, t12_y_test, t21_y_test), axis=0), y_pred_proba_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

# Plot ROC curve for RandomForest
plt.figure(figsize=(5, 5))
plt.plot(fpr_rf, tpr_rf, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_rf)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("Receiver Operating Characteristic Using RandomForest")
plt.legend(loc="lower right")
plt.show()

# Calculate accuracy for RandomForest
score_rf = accuracy_score(np.concatenate((t1_y_test, t12_y_test, t21_y_test), axis=0), y_pred_rf)
print(f"Accuracy score using RandomForest: {score_rf}")
