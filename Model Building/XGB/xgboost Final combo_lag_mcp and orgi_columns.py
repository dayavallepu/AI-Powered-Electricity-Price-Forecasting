import pandas as pd  # for data manipulation
import numpy as np  # for numerical calculations
import matplotlib.pyplot as plt  # for data visualizations
import xgboost as xgb  # for XGBoost model
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error  # for model evaluation
from sqlalchemy import create_engine  # for interacting with databases
from urllib.parse import quote  # for URL encoding
from sklearn.preprocessing import RobustScaler  # for feature scaling
import os  # for operating system interactions
import joblib  # for saving and loading the model

# Change directory to save the model
os.chdir(r'D:\360 DigiTMG projects\Project-2\AI-Powered Electricity Price Forecasting\9. Model Building\XGB')

# Database connection
user = 'root'  # Username for MySQL
pw = quote('Daya@123')  # Encode password for MySQL connection
db = 'ele_price_forecast'  # Database name
engine = create_engine(f'mysql+pymysql://{user}:{pw}@localhost/{db}')  # Create database connection

# Load data
df = pd.read_sql('SELECT * FROM ele_price_forecast.ele_price_forecast_preprocessed_lstm;', con=engine)  # Load data into a DataFrame
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

# Features & Target
X = df.drop(columns=['MCP', 'Datetime']).values  # Features
y = df['MCP'].values  # Target

# Apply Robust Scaling
scaler = RobustScaler()  # Initialize RobustScaler
X_scaled = scaler.fit_transform(X)  # Fit and transform features

# Train XGBoost Model 
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',  # Objective function for regression
    n_estimators=200,  # Number of trees in the forest
    learning_rate=0.03,  # Learning rate
    max_depth=8,  # Maximum depth of the tree
    subsample=0.8,  # Subsample ratio of the training instances
    colsample_bytree=0.8,  # Subsample ratio of columns when constructing each tree
    reg_lambda=0.5,  # L2 regularization term on weights
    reg_alpha=0.1  # L1 regularization term on weights
)

xgb_model.fit(X_scaled, y)  # Fit the model

# Save Model & Scaler
joblib.dump(xgb_model, 'final_xgboost_model.pkl')  # Save the trained model to a file
joblib.dump(scaler, 'scaler.pkl')  # Save the scaler to a file

print("Model and Scaler Saved Successfully!")  # Print confirmation message

# Load the model and scaler for prediction
xgb_model = joblib.load('final_xgboost_model.pkl')  # Load the trained model
scaler = joblib.load('scaler.pkl')  # Load the scaler

# Future Forecasting
future_steps = 200  # Number of future steps to forecast
future_predictions = []  # List to store future predictions
upper_bound = []  # List to store upper bound of confidence interval
lower_bound = []  # List to store lower bound of confidence interval

# Use last known input as the starting point
last_known_lags = X_scaled[-1]  # Last known input
time_interval = df['Datetime'].iloc[-1] - df['Datetime'].iloc[-2]  # Time step interval
last_known_datetime = df['Datetime'].iloc[-1]  # Last known datetime

# 95% Confidence Interval
error_std = np.std(y - xgb_model.predict(X_scaled))  # Standard deviation of errors
confidence_interval = 1.96 * error_std  # 95% CI

future_dates = []  # List to store future dates

for i in range(future_steps):
    # Predict the next MCP
    next_pred = xgb_model.predict(last_known_lags.reshape(1, -1))[0]

    # Compute confidence intervals
    upper_ci = next_pred + confidence_interval
    lower_ci = next_pred - confidence_interval

    # Store predictions
    future_predictions.append(next_pred)
    upper_bound.append(upper_ci)
    lower_bound.append(lower_ci)

    # Generate next timestamp
    next_datetime = last_known_datetime + (i + 1) * time_interval
    future_dates.append(next_datetime)

    # Update lag features
    next_pred_scaled = scaler.transform(np.array([next_pred] * X.shape[1]).reshape(1, -1))  
    last_known_lags = np.roll(last_known_lags, -1)  # Shift window
    last_known_lags[-1] = next_pred_scaled[0, 0]  # Add new scaled prediction

# Convert results to DataFrame
future_df = pd.DataFrame({'Datetime': future_dates, 'Predicted_MCP': future_predictions,
                          'Upper_CI': upper_bound, 'Lower_CI': lower_bound})

# Save Forecast Data
future_df.to_csv('future_forecast.csv', index=False)  # Save forecast data to a CSV file

# Plot Future Forecast
plt.figure(figsize=(12, 6))
plt.plot(future_df['Datetime'], future_df['Predicted_MCP'], label='Predicted MCP', color='green', linestyle='dashed')
plt.fill_between(future_df['Datetime'], future_df['Lower_CI'], future_df['Upper_CI'], color='gray', alpha=0.3, label='95% CI')
plt.xlabel('Datetime')
plt.ylabel('MCP')
plt.xticks(rotation=45)
plt.legend()
plt.title('Future MCP Forecast for Next 200 Time Steps')
plt.show()
