import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_csv("/content/dataset_2020_2022.csv")
df['DateTime'] = pd.to_datetime(df['DateTime'])
df = df.sort_values(by='DateTime')
df.dropna(subset=['Temperature (\u00b0C)', 'Humidity', 'Value'], inplace=True)

# Handle outliers by interpolation
def handle_outliers_time_series(column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Identify outliers
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    if outliers.sum() > 0:
        print(f"Column '{column}' - Number of outliers: {outliers.sum()}")
        # Replace outliers with linear interpolation
        df.loc[outliers, column] = np.nan
        df[column] = df[column].interpolate(method='linear')
        print(f"Outliers in column '{column}' handled by interpolation.")
    else:
        print(f"No outliers found in column '{column}'.")

# Apply to the 'Value' column
handle_outliers_time_series('Value')

# Function to check for outliers
def check_outliers(column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Count outliers
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    print(f"Column '{column}' - Number of outliers: {len(outliers)}")
    if len(outliers) == 0:
        print(f"No outliers found in column '{column}'.")
    else:
        print(f"Outliers in column '{column}' need further review.")

# Verify outliers for relevant columns
check_outliers('Temperature (\u00b0C)')
check_outliers('Humidity')
check_outliers('Value')

