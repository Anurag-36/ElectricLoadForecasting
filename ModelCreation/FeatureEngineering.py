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

# Add temporal features
df['DayOfMonth'] = df['DateTime'].dt.day
df['DayOfWeek'] = df['DateTime'].dt.weekday  # Monday=0, Sunday=6
df['HourOfDay'] = df['DateTime'].dt.hour  # Add the HourOfDay feature
df['Quarter'] = df['DateTime'].dt.quarter
df['Month'] = df['DateTime'].dt.month
df['Year'] = df['DateTime'].dt.year
df['DayOfYear'] = df['DateTime'].dt.dayofyear
df['WeekOfYear'] = df['DateTime'].dt.isocalendar().week

from tabulate import tabulate

# Print the head and tail in tabular format
print("Head of DataFrame:")
print(tabulate(df.head(), headers='keys', tablefmt='pretty', showindex=False))

print("\nTail of DataFrame:")
print(tabulate(df.tail(), headers='keys', tablefmt='pretty', showindex=False))


import pandas as pd

# Calculate the correlation matrix for all features
correlation_matrix = df.corr()

# Print the correlation between each feature and the target 'Value'
print("Correlation with 'Value':")
print(correlation_matrix['Value'].sort_values(ascending=False))

import seaborn as sns

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f', vmin=-1, vmax=1)
plt.title('Correlation Heatmap ')
plt.show()