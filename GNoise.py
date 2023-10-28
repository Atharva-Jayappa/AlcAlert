import numpy as np
import pandas as pd

# Read your reshaped EEG data from the CSV file into a DataFrame
df = pd.read_csv('reshaped_eeg_data.csv')

# Define a small standard deviation for minimal noise
sigma = 0.01  # Adjust this value to control the amount of noise (smaller values = less noise)

# Generate minimal Gaussian noise
num_columns = df.shape[1] - 7  # Calculate the number of EEG columns (excluding non-EEG columns)
noise = np.random.normal(0, sigma, size=(df.shape[0], num_columns))  # Generate noise for all rows
# Apply noise to EEG columns
eeg_columns = df.columns[7:]  # Select only EEG columns
df[eeg_columns] += noise

# Save the DataFrame with minimal noise
df.to_csv('minimal_noise_eeg_data.csv', index=False)
