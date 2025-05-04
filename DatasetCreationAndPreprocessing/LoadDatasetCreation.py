import os
import pandas as pd

# Define the root folder where the year folders are located
root_folder = r'/content/SLDC_Data'  # Replace with the actual path to the root folder

# Initialize an empty list to collect data from all CSV files
all_data = []
# Walk through the directory structure
for year_folder in os.listdir(root_folder):
    year_folder_path = os.path.join(root_folder, year_folder)

    if os.path.isdir(year_folder_path):  # Check if it is a folder
        for month_folder in os.listdir(year_folder_path):
            month_folder_path = os.path.join(year_folder_path, month_folder)

            if os.path.isdir(month_folder_path):  # Check if it is a folder
                # Process each CSV file in the month folder
                for csv_file in os.listdir(month_folder_path):
                    if csv_file.endswith('.csv'):
                        file_path = os.path.join(month_folder_path, csv_file)

                        # Extract date from the file name (assuming format like "01-01-2020.csv")
                        date_part = csv_file.split('.')[0]  # Example: "01-01-2020"

                        # Clean the date part to remove any extra spaces and handle special characters
                        date_part = date_part.strip()

                        try:
                            # Convert "day month year" format to "year-month-day" for correct date parsing
                            day, month, year = date_part.split('-')
                            date_part = f"{year}-{month}-{day}"  # Convert to "yyyy-mm-dd"

                            # Load the CSV file (no headers, assume columns 'time' and 'value')
                            data = pd.read_csv(file_path, header=None, names=['Time', 'Value'], skip_blank_lines=True)
                            print(f"Loaded data from {csv_file}.")

                            # Debugging: Print first few rows of the 'Time' column to check for issues
                            print("First few rows of Time column:")
                            print(data['Time'].head())

                            # Clean up 'Time' column (remove any extra text like "time")
                            data['Time'] = data['Time'].str.replace('time', '').str.strip()

                            # Debugging: Print cleaned 'Time' column
                            print("Cleaned 'Time' column:")
                            print(data['Time'].head())

                            # Ensure there are no extra spaces in date_part
                            date_part = date_part.strip()

                            # Combine the date from the file name with the 'Time' column to create a 'DateTime' column
                            data['DateTime'] = pd.to_datetime(date_part + ' ' + data['Time'], format='%Y-%m-%d %H:%M', errors='coerce')

                            # Drop the original 'Time' column if you no longer need it
                            data.drop('Time', axis=1, inplace=True)

                            # Append the current data to the all_data list
                            all_data.append(data)
                        except Exception as e:
                            print(f"Error processing {csv_file}: {e}")
                            # Optionally, you can log the exact row causing issues for further debugging
                            with open('error_log.txt', 'a') as f:
                                f.write(f"Error in {csv_file}: {e}\n")

# Check if the list is empty before concatenating
if all_data:
    # Concatenate all dataframes into one
    final_data = pd.concat(all_data, ignore_index=True)

    # Optionally, sort the data by DateTime if needed
    final_data.sort_values(by='DateTime', inplace=True)

    # Save the final combined data to a new CSV file
    final_data.to_csv('combined_load_data.csv', index=False)

    print("Data has been successfully combined and saved.")
else:
    print("No data was found to concatenate. Please check the input files and paths.")
# Load the data
data = pd.read_csv(r'/content/combined_load_data.csv')
# Print the column names to verify
print("Columns in the dataset:", data.columns)
# Reorder columns to have DateTime first
data = data[['DateTime', 'Value']]

 # Save the cleaned data
data.to_csv(r'/content/combined_load_data_final.csv', index=False)
print("Data cleaned and saved successfully.")