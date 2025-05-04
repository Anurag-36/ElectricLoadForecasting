# COMBINING BOTH DATASET TO ONE
import pandas as pd

# Load the two CSV files
file1_path = '/content/interpolated_weather_data.csv'  # Replace with the path to your first file
file2_path = '/content/combined_load_data_final.csv'  # Replace with the path to your second file

data1 = pd.read_csv(file1_path)
data2 = pd.read_csv(file2_path)

# Ensure the DateTime column is in datetime format
data1['DateTime'] = pd.to_datetime(data1['DateTime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
data2['DateTime'] = pd.to_datetime(data2['DateTime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

# Merge the two datasets based on DateTime
combined_data = pd.merge(data1, data2, on='DateTime', how='outer', suffixes=('_file1', '_file2'))

# Sort by DateTime
combined_data = combined_data.sort_values(by='DateTime')

# Save the combined data to a new CSV file (optional)
combined_data.to_csv('/content/combined_BOTHdata.csv', index=False)

# Print the first few rows to verify
print(combined_data.head())

# CLEANING

import pandas as pd

# Assuming you have a DataFrame df with 'DateTime', 'Temperature (°C)', 'Humidity', and 'Value' columns
# Load your dataset (example)
df = pd.read_csv('/content/combined_data.csv')

# Convert 'DateTime' to pandas datetime format
df['DateTime'] = pd.to_datetime(df['DateTime'])

# Set 'DateTime' as the index for time-based operations
df.set_index('DateTime', inplace=True)

# Check for missing values in each column
missing_values = df.isnull().sum()
print("Missing values in each column:")
print(missing_values)

# Create a new DataFrame with missing values handled (using forward fill)
df_new = df.fillna(method='ffill')

# Alternatively, you can use interpolation instead of forward fill
# df_new = df.interpolate(method='linear')

# Check if there are any remaining missing values in the new dataset
missing_values_after = df_new.isnull().sum()
print("Missing values after filling/interpolating:")
print(missing_values_after)

# Save the corrected DataFrame to a new CSV file
df_new.to_csv('/content/corrected_dataset.csv')

print("New dataset with corrected values has been saved as 'corrected_dataset.csv'.")
# Load your dataset
df = pd.read_csv('/content/corrected_dataset.csv')

# Convert 'datetime' column to datetime format (if it's not already)
df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')

# Remove rows where 'DateTime' is missing
df_cleaned = df.dropna(subset=['DateTime'])

# Optionally, reset the index
df_cleaned.reset_index(drop=True, inplace=True)

# Save the cleaned dataset to a new CSV file
df_cleaned.to_csv('/content/cleaned_dataset.csv', index=False)

print(f"Rows with missing datetime have been removed. The cleaned dataset has {len(df_cleaned)} rows.")

# Define start and end dates
start_date = pd.to_datetime('2020-01-01 00:00:00')
end_date = pd.to_datetime('2022-12-31 23:55:00	')

# Generate a date range with 5-minute intervals
date_range = pd.date_range(start=start_date, end=end_date, freq='5T')

# Number of rows in the dataset
num_rows = len(date_range)

print(f"The dataset will have {num_rows} rows from 2020 to 2023 at 5-minute intervals.")

# FINDING MISSING DATA FRAMES
# Load your dataset
df = pd.read_csv('/content/cleaned_dataset.csv')

# Convert 'DateTime' column to datetime format
df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')

# Set 'DateTime' as index if it's not already
df.set_index('DateTime', inplace=True)

# Generate the expected date range with 5-minute intervals
start_date = pd.to_datetime('2020-01-01 00:00:00')
end_date = pd.to_datetime('2023-12-31 22:45:00')
expected_range = pd.date_range(start=start_date, end=end_date, freq='5T')

# Find missing DateTime entries by comparing the expected range to the existing data
missing_dates = expected_range.difference(df.index)

# Convert missing dates to a DataFrame or Series and save to CSV
missing_dates_df = pd.DataFrame(missing_dates, columns=['Missing DateTime'])
missing_dates_df.to_csv('missing_datetimes.csv', index=False)

print("Missing DateTime entries have been saved to 'missing_datetimes.csv'.")

#SPLITTING BECAUSE 2023 HAS A LOT OF MISSING FRAMES

import pandas as pd

# Load your dataset
df = pd.read_csv('/content/cleaned_dataset.csv')

# Convert 'DateTime' column to datetime format
df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')

# Set 'DateTime' as the index
df.set_index('DateTime', inplace=True)

# Filter and save the 2020-2022 data
df_2020_2022 = df.loc['2020-01-01':'2022-12-31']
df_2020_2022.to_csv('/content/dataset_2020_2022.csv')

# Filter and save the 2023 data
df_2023 = df.loc['2023-01-01':'2023-12-31']
df_2023.to_csv('/content/dataset_2023.csv')

print("Two separate files have been saved: 'dataset_2020_2022.csv' and 'dataset_2023.csv'.")

