import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from bs4 import BeautifulSoup
import re
import time
import csv

# Initialize WebDriver
driver = webdriver.Chrome()

def fetch_weather_data(date, month, year):
    # Update the URL to include the current month and year
    url = f'https://www.timeanddate.com/weather/india/new-delhi/historic?month={month}&year={year}'

    # Open the webpage
    driver.get(url)

    # Wait for the page to load completely
    time.sleep(3)

    # Select the date from the dropdown
    select_element = driver.find_element(By.ID, 'wt-his-select')
    select = Select(select_element)

    # Check if the desired date is in the dropdown options
    available_dates = [option.text for option in select.options]
    if date not in available_dates:
        print(f"Date {date} is not available in the dropdown. Skipping.")
        return []

    # Select the desired date
    select.select_by_visible_text(date)

    # Wait for the content to update
    time.sleep(3)

    # Fetch the updated page content
    page_content = driver.page_source

    # Parse the content with BeautifulSoup
    soup = BeautifulSoup(page_content, 'html.parser')
    table = soup.find('table', {'id': 'wt-his'})  # Ensure table ID is correct

    if not table:
        print(f"No table found for date: {date}")
        return []

    data = []

    rows = table.find_all('tr')
    for row in rows[1:]:  # Skip header row
        time_cell = row.find('th')  # Time is in the <th> tag

        if time_cell:
            # Extract time part only, ignoring content inside <span>
            time_text = time_cell.contents[0].strip() if time_cell.contents else ''
            if not any(char.isdigit() for char in time_text):
                continue

        cells = row.find_all('td')

        if len(cells) >= 5:
            time_value = time_text.replace('.', ':')
            temp = cells[1].text.strip()
            temp = re.sub(r'[^\d.]', '', temp) + '°C'
            humidity = cells[5].text.strip()
            data.append([time_value, temp, humidity])

    return data

def save_to_csv(date, data):
    # Extract the month and year for folder creation
    month, year = date.split()[1], date.split()[2]
    folder_path = f"{year}/{month}"

    # Create directories if they do not exist
    os.makedirs(folder_path, exist_ok=True)

    # Create the file name using the date
    file_name = f"{folder_path}/{date.replace(' ', '_')}.csv"

    # Open CSV file and write data
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Time", "Temperature (°C)", "Humidity"])  # Write headers
        writer.writerows(data)  # Write the actual data

    print(f"Data for {date} saved to {file_name}")

# Generate dates for the whole year (example: 2020)
year = 2023
months = [
    "January", "February", "March", "April", "May",
    "June", "July", "August", "September", "October",
    "November", "December"
]

# List of days for each month (taking into account leap year for February)
days_in_month = {
    "January": 31, "February": 29 if year % 4 == 0 else 28, "March": 31,
    "April": 30, "May": 31, "June": 30,
    "July": 31, "August": 31, "September": 30,
    "October": 31, "November": 30, "December": 31
}

for month_index, month in enumerate(months, start=1):
    for day in range(1, days_in_month[month] + 1):
        date = f"{day} {month} {year}"
        print(f"Fetching data for {date}...")
        weather_data = fetch_weather_data(date, month_index, year)
        if weather_data:
            save_to_csv(date, weather_data)

# Close the browser
driver.quit()  # Make sure to close the WebDriver at the end