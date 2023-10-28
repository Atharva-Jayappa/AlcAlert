import pandas as pd
import numpy as np

df_original = pd.read_csv('C:\\Users\\jatha\\Desktop\\Sem5\\EDI\\SMNI_CMI_TEST\\Data1.csv')

# Pivot the data to create a 64 by 256 format
df_reshaped = df_original.pivot(
    index=['trial number', 'sensor position', 'subject identifier', 'matching condition', 'channel', 'name', 'time'],
    columns='sample num',
    values='sensor value')

df_reshaped.reset_index(inplace=True)

df_reshaped.to_csv('reshaped_eeg_data.csv', index=False)
