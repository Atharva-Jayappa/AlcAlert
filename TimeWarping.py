import pandas as pd
import numpy as np

# Load the original data
df_original = pd.read_csv('C:\\Users\\jatha\\Desktop\\Sem5\\EDI\\SMNI_CMI_TEST\\Data1.csv')

# Pivot the data to create a 64 by 256 format
df_reshaped = df_original.pivot(
    index=['trial number', 'sensor position', 'subject identifier', 'matching condition', 'channel', 'name', 'time'],
    columns='sample num',
    values='sensor value')

df_reshaped.reset_index(inplace=True)

# Define the time warping factor (e.g., 0.8 for 80% of the original duration)
time_warping_factor = 0.8  # Adjust this value as needed

# Apply time warping by interpolating the data
interpolated_data = []
for time, group in df_reshaped.groupby('time'):
    new_time = np.arange(0, len(group) * time_warping_factor, time_warping_factor)
    interpolated_group = group.interpolate(method='linear', limit_direction='both')
    interpolated_group['time'] = new_time
    interpolated_data.append(interpolated_group)

# Concatenate the interpolated data back into a DataFrame
df_reshaped_warped = pd.concat(interpolated_data)

# Save the time-warped data
df_reshaped_warped.to_csv('reshaped_eeg_data_warped.csv', index=False)
