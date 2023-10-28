import numpy as np
import pandas as pd
import os
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras

seed = 123
random.seed = seed

# Define the path to your training and test data directories
train_dir = 'C:/Users/jatha/Desktop/Sem5/EDI/SMNI_CMI_TRAIN/Train'
test_dir = 'C:/Users/jatha/Desktop/Sem5/EDI/SMNI_CMI_TEST/Test'
classifier = {'a': 1, 'c': 0}

# Load and preprocess EEG data for a specific condition
def process_data(data_dir, condition, data_size):
    X = np.zeros((data_size, 256, 64))
    y = np.zeros(data_size)
    count = 0

    filenames_list = os.listdir(data_dir)
    random.shuffle(filenames_list)  # Shuffle the files for randomness

    for file_name in tqdm(filenames_list):
        temp_df = pd.read_csv(os.path.join(data_dir, file_name))

        if temp_df["matching condition"][0] == condition:
            sensor_values = temp_df["sensor value"].apply(lambda x: np.fromstring(x[1:-1], sep=" ", dtype=np.float32) if isinstance(x, str) else np.array([x], dtype=np.float32))
            X[count] = sensor_values.values.reshape(256, 64)
            y[count] = classifier[temp_df['subject identifier'][0]]
            count += 1

        if count >= data_size:
            break

    return X, y

# Create training and testing datasets for each condition
train_size = 465
test_size = train_size

# Condition 'S1 obj'
s1_X_train, s1_y_train = process_data(train_dir, 'S1 obj', train_size)
t1_X_test, t1_y_test = process_data(test_dir, 'S1 obj', test_size)

# Condition 'S2 match'
s2_X_train, s2_y_train = process_data(train_dir, 'S2 match', train_size)
t2_X_test, t2_y_test = process_data(test_dir, 'S2 match', test_size)

# Standardize the input data
scaler = StandardScaler()
s1_X_train = np.array([scaler.fit_transform(x) for x in s1_X_train])
t1_X_test = np.array([scaler.transform(x) for x in t1_X_test])
s2_X_train = np.array([scaler.fit_transform(x) for x in s2_X_train])
t2_X_test = np.array([scaler.transform(x) for x in t2_X_test])

# Define and train the CNN model for 'S1 obj'
model_s1 = keras.Sequential([
    keras.layers.Reshape(target_shape=(256, 64, 1), input_shape=(256, 64)),
    keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model_s1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_s1 = model_s1.fit(s1_X_train, s1_y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model for 'S1 obj' on the test set
test_loss_s1, test_accuracy_s1 = model_s1.evaluate(t1_X_test, t1_y_test)
print("Condition 'S1 obj' Test Loss:", test_loss_s1)
print("Condition 'S1 obj' Test Accuracy:", test_accuracy_s1)

# Define and train the CNN model for 'S2 match'
model_s2 = keras.Sequential([
    keras.layers.Reshape(target_shape=(256, 64, 1), input_shape=(256, 64)),
    keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model_s2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_s2 = model_s2.fit(s2_X_train, s2_y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model for 'S2 match' on the test set
test_loss_s2, test_accuracy_s2 = model_s2.evaluate(t2_X_test, t2_y_test)
print("Condition 'S2 match' Test Loss:", test_loss_s2)
print("Condition 'S2 match' Test Accuracy:", test_accuracy_s2)


# Plot training history (loss and accuracy over epochs) for 'S1 obj'
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history_s1.history['loss'], label='Training Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title("S1 obj Model Loss")

plt.subplot(1, 2, 2)
plt.plot(history_s1.history['accuracy'], label='Training Accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title("S1 obj Model Accuracy")

plt.show()

# Plot training history (loss and accuracy over epochs) for 'S2 match'
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history_s2.history['loss'], label='Training Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title("S2 match Model Loss")

plt.subplot(1, 2, 2)
plt.plot(history_s2.history['accuracy'], label='Training Accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title("S2 match Model Accuracy")

plt.show()
