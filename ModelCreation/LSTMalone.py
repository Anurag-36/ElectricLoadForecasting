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

# Split the data into training and testing
train_data = df[df['DateTime'] < '2022-05-30']
test_data = df[df['DateTime'] >= '2022-06-01']

# Normalize the features and target for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))

selected_features = [
    'Temperature (°C)', 'Humidity', 'Year', 'HourOfDay', 'Quarter', 'Month', 'DayOfYear', 'WeekOfYear'
]
train_scaled = train_data.copy()
test_scaled = test_data.copy()
train_scaled[selected_features + ['Value']] = scaler.fit_transform(train_data[selected_features + ['Value']])
test_scaled[selected_features + ['Value']] = scaler.transform(test_data[selected_features + ['Value']])

# Prepare sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length, :-1])  # Features (exclude target)
        y.append(data[i + seq_length, -1])  # Target (next value)
    return np.array(X), np.array(y)


seq_length = 12
features_train = train_scaled[selected_features].values
target_train = train_scaled['Value'].values
features_test = test_scaled[selected_features].values
target_test = test_scaled['Value'].values

X_train, y_train = create_sequences(np.column_stack([features_train, target_train]), seq_length)
X_test, y_test = create_sequences(np.column_stack([features_test, target_test]), seq_length)

from tensorflow.keras.regularizers import l2  # Import L2 regularizer

# Define the LSTM model
lstm_model = Sequential()

# Layer 1: LSTM Layer with L2 Regularization
#  - Units: Number of neurons in the LSTM layer (64)
#  - Return sequences: True, because we are adding another LSTM layer
lstm_model.add(LSTM(
    units=64,
    return_sequences=True,
    input_shape=(X_train.shape[1], X_train.shape[2]),
    kernel_regularizer=l2(0.001)  # L2 regularization with strength 0.001
))
# Dropout Layer to reduce overfitting
lstm_model.add(Dropout(0.3))  # Drops 20% of neurons during training

# Layer 2: Second LSTM Layer with L2 Regularization
#  - Units: Number of neurons in the LSTM layer (32)
#  - Return sequences: False, because this is the last LSTM layer
lstm_model.add(LSTM(
    units=32,
    return_sequences=False,
    kernel_regularizer=l2(0.001)  # L2 regularization with strength 0.001
))
# Dropout Layer to reduce overfitting
lstm_model.add(Dropout(0.03))

# Layer 3: Dense Layer (Output Layer) with L2 Regularization
#  - Units: 1, because we are predicting a single value (the target variable)
lstm_model.add(Dense(
    units=1,
    kernel_regularizer=l2(0.001)  # L2 regularization with strength 0.001
))

# Compile the model
#  - Optimizer: Adam (adaptive learning rate optimization)
#  - Loss: Mean Squared Error (used for regression problems)
lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

# Train the model
history = lstm_model.fit(
    X_train, y_train,
    epochs=20,  # Number of passes through the dataset
    batch_size=32,  # Reduced batch size for better generalization
    validation_data=(X_test, y_test),
    callbacks=[early_stopping],  # Stops training if validation loss doesn't improve
    verbose=1
)


# Make predictions with LSTM
lstm_train_predictions = lstm_model.predict(X_train).flatten()
lstm_test_predictions = lstm_model.predict(X_test).flatten()


train_predictions = lstm_model.predict(X_train)
test_predictions = lstm_model.predict(X_test)

# Rescale the predictions and actual values (since you're working with scaled data)
train_predictions_rescaled = scaler.inverse_transform(np.column_stack([features_train[seq_length:], train_predictions]))[:, -1]
test_predictions_rescaled = scaler.inverse_transform(np.column_stack([features_test[seq_length:], test_predictions]))[:, -1]

train_y_rescaled = scaler.inverse_transform(np.column_stack([features_train[seq_length:], y_train]))[:, -1]
test_y_rescaled = scaler.inverse_transform(np.column_stack([features_test[seq_length:], y_test]))[:, -1]

# Calculate evaluation metrics for train and test data

# Train MAPE and Accuracy
train_mape = np.mean(np.abs((train_y_rescaled - train_predictions_rescaled) / train_y_rescaled)) * 100
train_accuracy = 100 - train_mape  # Accuracy = 100% - MAPE

# Test MAPE and Accuracy
test_mape = np.mean(np.abs((test_y_rescaled - test_predictions_rescaled) / test_y_rescaled)) * 100
test_accuracy = 100 - test_mape  # Accuracy = 100% - MAPE

# Print the results
print(f"Train MAPE: {train_mape:.2f}%")
print(f"Train Accuracy: {train_accuracy:.2f}%")

print(f"Test MAPE: {test_mape:.2f}%")
print(f"Test Accuracy: {test_accuracy:.2f}%")


# Saving the LSTM model to a file (HDF5 format)
lstm_model.save('/content/lstm_model_hybv3.h5')