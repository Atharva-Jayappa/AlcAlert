import os
import pandas as pd
from collections import defaultdict

# Define the path to the 'SMNI_CMI_TRAIN' directory
train_dir = 'C:/Users/jatha/Desktop/Sem5/EDI/SMNI_CMI_TRAIN/Train'

# Initialize dictionaries to count cases within 'a' and 'c' categories
cases_a = defaultdict(int)
cases_c = defaultdict(int)

# Iterate through the files in the directory
for root, _, files in os.walk(train_dir):
    for file_name in files:
        file_path = os.path.join(root, file_name)

        # Check if the file is a CSV file
        if file_name.endswith('.csv'):
            # Read the CSV file
            df = pd.read_csv(file_path)

            # Check the 'subject identifier' column
            subject_identifier = df['subject identifier'][0]

            # Count cases within 'a' and 'c' categories based on 'subject identifier'
            if subject_identifier == 'a':
                condition = df['matching condition'][0]
                cases_a[condition] += 1
            elif subject_identifier == 'c':
                condition = df['matching condition'][0]
                cases_c[condition] += 1

# Print the counts for 'a' cases
print("Counts for 'a' cases:")
for condition, count in cases_a.items():
    print(f"'a' case '{condition}': {count}")

# Print the counts for 'c' cases
print("\nCounts for 'c' cases:")
for condition, count in cases_c.items():
    print(f"'c' case '{condition}': {count}")
