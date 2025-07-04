# importing required libraries
import pandas as pd  # for data manipulation
import numpy as np  # for numerical calculations
import matplotlib.pyplot as plt  # for data visualizations
import os  # for operating system interactions

# Disable OneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf  # for deep learning
from tensorflow.keras.models import Sequential  # for creating sequential models
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Dropout, Bidirectional  # for LSTM and other layers
from tensorflow.keras.optimizers import AdamW  # for AdamW optimizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler  # for feature scaling
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # for callbacks during training
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error  # for model evaluation
import seaborn as sns  # for data visualization
import joblib # for saving the models


# Load data
df = pd.read_excel(r"/content/drive/MyDrive/Copy of ele_price_forecast_preprocessed_LSTM.xlsx")  # Load data into a DataFrame
df.info()  # Display information about the DataFrame

# Ensure 'Datetime' column is in the correct format
df['Datetime'] = pd.to_datetime(df['Datetime'])  # Convert 'Datetime' column to datetime format

df.sort_values('Datetime', inplace=True)  # Sort data by 'Datetime'
df.head()  # Display the first few rows of the DataFrame

# Creating lag features
lag = 96  # Number of lag features
for i in range(1, lag + 1):
    df[f'MCP_lag_{i}'] = df['MCP'].shift(i)  # Create lag features for 'MCP'
df.dropna(inplace=True)  # Drop rows with NaN values

df.info()  # Display information about the DataFrame after creating lag features
df.isnull().sum()  # Check for any remaining null values
df.head(15) # Display the first 15 rows of the DataFrame

# Normalize features
scaler_features = RobustScaler()
scaled_features = scaler_features.fit_transform(df.drop(columns=['MCP', 'Datetime']))

scaler_target = RobustScaler()
df['MCP'] = scaler_target.fit_transform(df[['MCP']])

# Prepare data for LSTM
X_full, y_full = np.array(scaled_features), np.array(df[['MCP']].values)

# Reshape input for LSTM
X_full = X_full.reshape(X_full.shape[0], X_full.shape[1], 1)

# Build LSTM model
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
model = Sequential([
    LSTM(150, activation='tanh', return_sequences=True, input_shape=(X_full.shape[1], 1)),
    BatchNormalization(),
    LSTM(100, activation='tanh', return_sequences=True),
    LSTM(50, activation='tanh'),
    Dense(1)
])


optimizer = AdamW(learning_rate=0.01, clipnorm=1.0)  # Increase LR from 0.001 to 0.01
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=1e-5)
model.compile(optimizer=optimizer, loss='mse')
history = model.fit(X_full, y_full, epochs=25, batch_size=128,
                    verbose=1, callbacks=[reduce_lr])


# Forecast next 200 values
future_predictions = []
future_dates = [df['Datetime'].iloc[-1] + pd.Timedelta(minutes=15 * i) for i in range(1, 201)]
last_input = X_full[-1].reshape(1, X_full.shape[1], 1)

for _ in range(200):
    pred = model.predict(last_input)
    future_predictions.append(pred[0, 0])
    last_input = np.roll(last_input, -1)
    last_input[0, -1, 0] = pred[0, 0]

# Inverse transform forecasted values
future_predictions_inv = scaler_target.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Calculate 95% confidence interval
std_dev = np.std(future_predictions_inv)
ci_upper_forecast = future_predictions_inv.flatten() + 1.96 * std_dev
ci_lower_forecast = future_predictions_inv.flatten() - 1.96 * std_dev

# Create Forecast DataFrame
forecast_df = pd.DataFrame({
    'Datetime': future_dates,
    'Forecast': future_predictions_inv.flatten(),
    'CI_Lower': ci_lower_forecast,
    'CI_Upper': ci_upper_forecast
})

print(forecast_df)

# Plot the forecast with annotations
plt.figure(figsize=(12, 6))
plt.plot(forecast_df['Datetime'], forecast_df['Forecast'], label='Forecasted MCP', linestyle='dashed', color='green')
plt.fill_between(forecast_df['Datetime'], forecast_df['CI_Lower'], forecast_df['CI_Upper'],
                 color='gray', alpha=0.3, label='95% Confidence Interval')

# Annotate every tenth point
for i in range(0, len(forecast_df), 10):
    plt.annotate(f"{forecast_df['Forecast'].iloc[i]:.2f}",
                 (forecast_df['Datetime'].iloc[i], forecast_df['Forecast'].iloc[i]),
                 textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8, color='black')

plt.legend()
plt.title('Future MCP Forecast (Next 200 Predictions)')
plt.xlabel('Datetime')
plt.ylabel('MCP')
plt.xticks(rotation=45)
plt.show()

#save the model
model.save('final_lstm_combo_model.h5')
joblib.dump(scaler_features, 'scaler_features.pkl')
joblib.dump(scaler_target, 'scaler_target.pkl')
print("model saved successfully...")