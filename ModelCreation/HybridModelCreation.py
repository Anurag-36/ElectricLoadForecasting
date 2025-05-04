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

# XGBoost Model
xgb_model = XGBRegressor(
    n_estimators=500,                # Number of boosting rounds (trees)
    learning_rate=0.01,              # Learning rate
    max_depth=10,                      # Maximum depth of each tree
    early_stopping_rounds=15,         # Early stopping if no improvement after 50 rounds
    subsample=0.7,                    # Fraction of training data used per tree
    random_state=42,                  # Random state for reproducibility
    colsample_bytree=0.7,             # Fraction of features used for each tree
    reg_lambda=10,                     # L2 regularization term
    reg_alpha=1,                      # L1 regularization term
    gamma=15,
    eval_metric='mae',
    n_jobs=1
)

xgb_model.fit(train_data[selected_features], train_data['Value'],
        eval_set=[(train_data[selected_features], train_data['Value']), (test_data[selected_features], test_data['Value'])],

        verbose=True)


# Saving the XGBoost model to a file
xgb_model.save_model('/content/xgb_model_hybV3.json')

# Saving the LSTM model to a file (HDF5 format)
lstm_model.save('/content/lstm_model_hybv3.h5')
lstm_model.save('/content/lstm_model_hybv3.keras')

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from tensorflow.keras.models import load_model
import xgboost as xgb
from xgboost import DMatrix

# Load the saved LSTM model
lstm_model = load_model('/content/lstm_model_hybv3.keras')

# Load the saved XGBoost model
xgb_model = xgb.Booster()
xgb_model.load_model('/content/xgb_model_hybV3.json')


# Load LSTM and XGBoost predictions
# Example: Predict using the XGBoost model
# Create DMatrix for prediction
dtrain = DMatrix(train_data[selected_features])
dtest = DMatrix(test_data[selected_features])

# Predict with XGBoost
xgb_train_predictions = xgb_model.predict(dtrain)
xgb_test_predictions = xgb_model.predict(dtest)
# Make predictions with LSTM
lstm_train_predictions = lstm_model.predict(X_train).flatten()
lstm_test_predictions = lstm_model.predict(X_test).flatten()
# Ensure predictions have the same length
min_len_train = min(len(lstm_train_predictions), len(xgb_train_predictions))
min_len_test = min(len(lstm_test_predictions), len(xgb_test_predictions))

lstm_train_predictions = lstm_train_predictions[:min_len_train]
xgb_train_predictions = xgb_train_predictions[:min_len_train]
lstm_test_predictions = lstm_test_predictions[:min_len_test]
xgb_test_predictions = xgb_test_predictions[:min_len_test]

# Stack the predictions as features
stacked_train_predictions = np.column_stack([lstm_train_predictions, xgb_train_predictions])
stacked_test_predictions = np.column_stack([lstm_test_predictions, xgb_test_predictions])

# Initialize Random Forest Regressor
final_model = RandomForestRegressor(
    n_estimators=500,  # Number of trees
    max_depth=10,    # Let the model decide depth
    random_state=42,   # Reproducibility

    verbose=True
)

# Train the model
final_model.fit(stacked_train_predictions, y_train)


# Make final predictions
final_train_predictions = final_model.predict(stacked_train_predictions)
final_test_predictions = final_model.predict(stacked_test_predictions)

# Rescale predictions back to the original scale
train_actual = scaler.inverse_transform(np.column_stack([features_train[seq_length:], y_train]))[:, -1]
test_actual = scaler.inverse_transform(np.column_stack([features_test[seq_length:], y_test]))[:, -1]

final_train_predictions_rescaled = scaler.inverse_transform(
    np.column_stack([features_train[seq_length:], final_train_predictions])
)[:, -1]
final_test_predictions_rescaled = scaler.inverse_transform(
    np.column_stack([features_test[seq_length:], final_test_predictions])
)[:, -1]

# Calculate Evaluation Metrics
train_mae = mean_absolute_error(train_actual, final_train_predictions_rescaled)
test_mae = mean_absolute_error(test_actual, final_test_predictions_rescaled)

train_mse = mean_squared_error(train_actual, final_train_predictions_rescaled)
test_mse = mean_squared_error(test_actual, final_test_predictions_rescaled)

train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)

train_r2 = r2_score(train_actual, final_train_predictions_rescaled)
test_r2 = r2_score(test_actual, final_test_predictions_rescaled)

train_mape = np.mean(np.abs((train_actual - final_train_predictions_rescaled) / train_actual)) * 100
test_mape = np.mean(np.abs((test_actual - final_test_predictions_rescaled) / test_actual)) * 100

# Print the evaluation metrics
print(f"Train MAE: {train_mae:.4f}")
print(f"Test MAE: {test_mae:.4f}")
print(f"Train MSE: {train_mse:.4f}")
print(f"Test MSE: {test_mse:.4f}")
print(f"Train RMSE: {train_rmse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")
print(f"Train R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")
print(f"Train MAPE: {train_mape:.4f}")
print(f"Test MAPE: {test_mape:.4f}")

# Convert MAPE to accuracy
train_accuracy = 100 - train_mape
test_accuracy = 100 - test_mape
print(f"Train Accuracy: {train_accuracy:.2f}%")
print(f"Test Accuracy: {test_accuracy:.2f}%")


import joblib

# Save the trained RandomForest model
joblib.dump(final_model, '/content/random_forest_modelHybv3.pkl')


import matplotlib.pyplot as plt

# Plot for training set
plt.figure(figsize=(10, 6))
plt.plot(train_actual, label='Actual Train', color='blue', linestyle='-', linewidth=2)
plt.plot(final_train_predictions_rescaled, label='Predicted Train', color='red', linestyle='--', linewidth=2)

# Adding labels and title
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.title('Train Data: Predicted vs Actual')

# Show legend
plt.legend()

# Display the plot
plt.tight_layout()
plt.show()

# Plot for testing set
plt.figure(figsize=(10, 6))
plt.plot(test_actual, label='Actual Test', color='green', linestyle='-', linewidth=2)
plt.plot(final_test_predictions_rescaled, label='Predicted Test', color='orange', linestyle='--', linewidth=2)

# Adding labels and title
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.title('Test Data: Predicted vs Actual')

# Show legend
plt.legend()

# Display the plot
plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Define the range of time steps you want to visualize (200 to 300)
start_step = 50000
end_step = 52000

# Extract the actual and predicted values within the specified range
test_actual_range = test_actual[start_step:end_step+1]  # +1 to include the end step
test_predictions_range = final_test_predictions_rescaled[start_step:end_step+1]

# Plot for testing set in the range 200-300
plt.figure(figsize=(10, 6))
plt.plot(range(start_step, end_step+1), test_actual_range, label='Actual Test' , color='green', marker='o', linestyle='-', linewidth=2)
plt.plot(range(start_step, end_step+1), test_predictions_range, label='Predicted ', color='orange', marker='x', linestyle='--', linewidth=2)

# Adding labels and title
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.title('Test Data : Predicted vs Actual')

# Show legend
plt.legend()

# Display the plot
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt

# Define the range of time steps you want to visualize (50000 to 52000)
start_step = 50000
end_step = 52000

# Extract actual and predicted values in the specified range
test_actual_range = test_actual[start_step:end_step + 1]
test_predictions_range = final_test_predictions_rescaled[start_step:end_step + 1]

# Extract the corresponding DateTime values for the time step range
dates = test_data['DateTime'].iloc[start_step:end_step + 1]

# Plot for the test set in the range 50000 to 52000
plt.figure(figsize=(12, 6))
plt.plot(dates, test_actual_range, label='Actual Test', color='green', marker='o', linestyle='-', linewidth=2)
plt.plot(dates, test_predictions_range, label='Predicted', color='orange', marker='x', linestyle='--', linewidth=2)

# Adding labels and title
plt.xlabel('Date/Time')
plt.ylabel('Value')
plt.title('Test Data: Predicted vs Actual')

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45)

# Show legend
plt.legend()

# Display the plot
plt.tight_layout()
plt.show()

