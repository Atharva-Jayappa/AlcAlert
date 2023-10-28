import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import os

# Define the path to the 'SMNI_CMI_TRAIN' directory
data_dir = 'C:/Users/jatha/Desktop/Sem5/EDI/SMNI_CMI_TRAIN'

# Initialize lists to store data and labels
X = []
y = []

# Iterate through the files in the 'SMNI_CMI_TRAIN' directory
for root, _, files in os.walk(data_dir):
    for file_name in files:
        file_path = os.path.join(root, file_name)

        # Check if the file is a CSV file
        if file_name.endswith('.csv'):
            # Read the CSV file
            df = pd.read_csv(file_path)

            # Check the 'subject identifier' column
            subject_identifier = df['subject identifier'][0]

            # Extract sensor values
            sensor_values = df['sensor value'].apply(lambda x: np.fromstring(x[1:-1], sep=" ", dtype=np.float32)
            if isinstance(x, str) else np.array([x], dtype=np.float32))
            X.extend(sensor_values.values.reshape(-1, 256, 64))
            y.extend([subject_identifier] * len(sensor_values))

# Convert lists to NumPy arrays
X = np.array(X)
y = np.array(y)

# Standardize the input data
scaler = StandardScaler()
X = np.array([scaler.fit_transform(x) for x in X])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train the CNN model
model = keras.Sequential([
    keras.layers.Reshape(target_shape=(256, 64, 1), input_shape=(256, 64)),
    keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model on the entire dataset
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
