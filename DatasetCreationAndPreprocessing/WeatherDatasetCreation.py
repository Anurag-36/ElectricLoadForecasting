import os
import pandas as pd
from datetime import datetime

# MERGE WEATHER DATA

# Set the root directory where the year folders are stored
root_dir = r'/content/WeatherData'  # Replace with your directory

# Initialize an empty list to store the dataframes
dataframes = []
# Loop through each year folder
for year_folder in os.listdir(root_dir):
    year_path = os.path.join(root_dir, year_folder)
    if os.path.isdir(year_path):  # Check if it's a directory
        # Loop through each month folder within the year
        for month_folder in os.listdir(year_path):
            month_path = os.path.join(year_path, month_folder)
            if os.path.isdir(month_path):  # Check if it's a directory
                # Loop through each file in the month folder
                for filename in os.listdir(month_path):
                    if filename.endswith('.csv'):  # Assuming the files are in CSV format
                        file_path = os.path.join(month_path, filename)

                        # Load the CSV file into a dataframe with specified encoding
                        df = pd.read_csv(file_path, encoding='ISO-8859-1')

                        # Extract date from filename and convert it to DateTime format
                        date_str = filename.split('.')[0]  # e.g., "1_April_2020"
                        date_obj = datetime.strptime(date_str, '%d_%B_%Y')
                        df['Date'] = date_obj.date()

                        # Combine Date and Time columns to form a DateTime column
                        df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'])

                        # Drop the original Date and Time columns if needed
                        df.drop(columns=['Date', 'Time'], inplace=True)

                        # Append the dataframe to the list
                        dataframes.append(df)

# Concatenate all dataframes into one
all_weather_data = pd.concat(dataframes, ignore_index=True)

# Sort by DateTime for a continuous timeline
all_weather_data = all_weather_data.sort_values(by='DateTime').reset_index(drop=True)

# Save to a single CSV file
all_weather_data.to_csv('merged_weather_data.csv', index=False)



# REORDER THE WEATHER DATA

# Load the data
data = pd.read_csv(r'/content/merged_weather_data.csv')
# Print the column names to verify
print("Columns in the dataset:", data.columns)
if 'Temperature (°C)' in data.columns and 'Humidity' in data.columns:
    data['Temperature (°C)'] = data['Temperature (°C)'].str.replace('°C', '').replace('', float('nan')).astype(float)
    data['Humidity'] = data['Humidity'].str.replace('%', '').replace('', float('nan')).astype(float)

    # Drop rows with NaN values
    data.dropna(subset=['Temperature (°C)', 'Humidity'], inplace=True)

    # Reorder columns to have DateTime first
    data = data[['DateTime', 'Temperature (°C)', 'Humidity']]

    # Save the cleaned data
    data.to_csv('cleaned_weather_data.csv', index=False)
    print("Data cleaned and saved successfully.")
else:
    print("Check if 'Temperature (°C)' and 'Humidity' exist in the dataset columns.")


# INTERPOATE WEATHER DATA TO MATCH THE 5min INTERVAL

# Load the dataset (replace with your actual file path)
data = pd.read_csv(r'/content/cleaned_weather_data.csv')  # Replace with the correct path
# Ensure 'DateTime' is a datetime object
data['DateTime'] = pd.to_datetime(data['DateTime'])

# Set 'DateTime' as the index for reindexing
data.set_index('DateTime', inplace=True)

# Create a complete date range for each day from 00:00 to 23:55 at 5-minute intervals
start_date = data.index.min().normalize()  # Start from midnight of the first day
end_date = data.index.max().normalize() + pd.Timedelta(days=1) - pd.Timedelta(minutes=5)  # End at 23:55 of the last day
# Create a complete datetime range with 5-minute frequency
complete_date_range = pd.date_range(start=start_date, end=end_date, freq='5T')

# Reindex the data to fill in missing times with NaN values
data_resampled = data.reindex(complete_date_range)

# Identify missing data (times where data is missing)
missing_times = data_resampled[data_resampled.isnull().any(axis=1)].index
if len(missing_times) > 0:
    print("Missing data for the following times:")
    print(missing_times)

# Interpolate the missing values (for both Temperature and Humidity)
data_resampled['Temperature (°C)'] = data_resampled['Temperature (°C)'].interpolate(method='linear')
data_resampled['Humidity'] = data_resampled['Humidity'].interpolate(method='linear')

# Adjust decimal points to 2 places
data_resampled['Temperature (°C)'] = data_resampled['Temperature (°C)'].round(2)
data_resampled['Humidity'] = data_resampled['Humidity'].round(2)

# Reset index so the DateTime column is back as a column
data_resampled.reset_index(inplace=True)
data_resampled.rename(columns={'index': 'DateTime'}, inplace=True)

# Save the cleaned and interpolated data to a new CSV file
data_resampled.to_csv('interpolated_weather_data.csv', index=False)

print("Data has been cleaned, interpolated, and saved successfully.")













