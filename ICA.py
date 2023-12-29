import mne
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import glob
from matplotlib import pyplot as plt

_dfs_list = []
for csv_filename in tqdm(glob.glob('C:/Users/jatha/Desktop/Sem5/EDI/SMNI_CMI_TRAIN/*.csv')):
    _dfs_list.append(pd.read_csv(csv_filename))
df = pd.concat(_dfs_list)
del (_dfs_list)
df = df.drop(['Unnamed: 0'], axis=1)

channel_list = list(set(df['sensor position']))
channel_list.sort()

channel_mapping = {
    'AFZ': 'AFz',
    'CPZ': 'CPz',
    'CZ': 'Cz',
    'FCZ': 'FCz',
    'FP1': 'Fp1',
    'FP2': 'Fp2',
    'FPZ': 'Fpz',
    'FZ': 'Fz',
    'OZ': 'Oz',
    'POZ': 'POz',
    'PZ': 'Pz',
}

channel_mapping_full = dict()

for ch in channel_list:
    if ch in channel_mapping:
        channel_mapping_full[ch] = channel_mapping[ch]
    else:
        channel_mapping_full[ch] = ch

channel_list_fixed = [channel_mapping_full[ch] for ch in channel_list]

df['sensor position'] = df['sensor position'].map(channel_mapping_full)

transposed_df_list = []

for group_df in tqdm(
        df.groupby(['name', 'trial number', 'matching condition', 'sensor position', 'subject identifier'])):
    _df = pd.DataFrame(group_df[1]['sensor value']).T
    df.columns = [f'sample{idx}' for idx in range(256)]
    _df['name'] = group_df[0][0]
    _df['trial number'] = group_df[0][1]
    _df['matching condition'] = group_df[0][2]
    _df['sensor position'] = group_df[0][3]
    _df['subject identifier'] = group_df[0][4]

    transposed_df_list.append(_df)

df = pd.concat(transposed_df_list)
df = df[[*df.columns[-5:], *df.columns[0:-5]]]
df = df.reset_index(drop=True)


def get_record_df(df, name, trial_number, matching_condition, channel_list):
    df_record = df[df['name'].eq(name) & df['trial number'].eq(trial_number) & df['matching condition'].eq(
        matching_condition)].set_index(['sensor position']).loc[channel_list]
    return df_record


df_record = get_record_df(df, 'co2a0000364', 0, 'S1 obj', channel_list_fixed)


def get_signal_array(df, name, trial_number, matching_condition, channel_list):
    df_record = get_record_df(df, name, trial_number, matching_condition, channel_list)
    return df_record.to_numpy()[:, 4:]


signal_array = get_signal_array(df, 'co2a0000364', 10, 'S1 obj', channel_list_fixed)

info = mne.create_info(ch_names=channel_list_fixed, sfreq=256, ch_types=['eeg'] * 64)
raw = mne.io.RawArray(signal_array, info)

standard_1020_montage = mne.channels.make_standard_montage('standard_1020');
raw.drop_channels(['X', 'Y', 'nd'])
raw.set_montage(standard_1020_montage)

raw_filtered = raw.copy().filter(8, 27, verbose=False);
raw_filtered.plot_psd();
raw_filtered.plot_psd(average=True);

ica = mne.preprocessing.ICA(random_state=42, n_components=20)
ica.fit(raw.copy().filter(1, None, verbose=False), verbose=False)
ica.plot_components()
