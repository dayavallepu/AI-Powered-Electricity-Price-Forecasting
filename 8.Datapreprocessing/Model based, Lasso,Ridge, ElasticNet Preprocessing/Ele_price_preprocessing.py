
#*Model-based Preprocessing*

# Importing the required libraries
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical computations
import matplotlib.pyplot as plt  # For plotting graphs
from sqlalchemy import create_engine  # For interacting with databases
from urllib.parse import quote # For encoding passwords that might have special characters
import os  # For file operations

user = 'root' # Database connection credentials
pw = quote('Daya@123') # Encode the password to handle special characters
db = 'ele_price_forecast' # Database name
engine = create_engine(f'mysql+pymysql://{user}:{pw}@localhost/{db}') # Create engine for database connection
# Load the data into a DataFrame called ele_price_forecast
sql = 'SELECT * FROM ele_price_forecast_raw;'
df = pd.read_sql(sql, con=engine) # Load the data into a DataFrame of date_time_MCP
df.head() # Display the first few rows of the DataFrame
df.info() # Display the column names and data types

# Rename target column for better readability
df.rename(columns={"MCP (Rs/MWh) *": "MCP"}, inplace=True)

# Selecting only required feature for model based approches
df2 = df[["Datetime", "MCP"]]
df2.info()


# Filter out zero or negative values in the MCP column
zero_values = df2[df2["MCP"] <= 0] # Filter out zero or negative values and Zero Values in the MCP column
zero_values # Display the zero or negative values in the MCP column
'''
we are cheecked the negative and zero values in the MCP column and we found that there are 0 values in the MCP column.
we checked the real time data from the IEX website at that time the values are available.
so we are going to impute the zerovalues with the Backward fill and Forward fill method. of the MCP column because the median is less sensitive to outliers than the mean.
                 Datetime  MCP
53860 2024-07-15 01:00:00  0.0
53861 2024-07-15 01:15:00  0.0
67662 2024-12-05 20:00:00  0.0
67663 2024-12-05 20:15:00  0.0
the above are the zero values in the MCP column.

'''
# Replace zero values with NaN
df2["MCP"] = df2["MCP"].replace(0, np.nan)

# Impute missing values using forward fill
df2["MCP"] = df2["MCP"].fillna(method="ffill")
# Impute missing values using backward fill
df2["MCP"] = df2["MCP"].fillna(method="bfill")

# Check for missing values in the dataframe
df2.isnull().sum()

# Check for any remaining zero or negative values in the MCP column
zero_values = df2[df2["MCP"] <= 0]
zero_values # Display the zero or negative values in the MCP column

# Sending imputed data to the database
df2.to_sql('Date_time_MCP_imputed', con = engine, if_exists = 'replace', index = False) 

# Load the data into a DataFrame called ele_price_forecast
sql = 'SELECT * FROM Date_time_MCP_imputed'
df3 = pd.read_sql(sql, con=engine) # Load the data into a DataFrame of date_time_MCP
df3.head() # Display the first few rows of the DataFrame    
df3.info() # Display the column names and data types

# Create new columns for time and MCP trends
df3["t"] = np.arange(1, 75700 + 1) # Create a new column 't' with sequential integers
df3["t_sq"] = df3["t"] * df3["t"] # Create a new column 't_sq' with the square of 't'
df3["log_MCP"] = np.log(df3["MCP"]) # Create a new column 'log_MCP' with the logarithm of 'MCP'

# exploring the datetime column and extracting the time of 15 min interval(assuming it's a string)
time = df3["Datetime"][0] # Extract the first element of the 'Datetime' column
time[14:16] # access the last two characters of the string

# creating new time columns for 15min interval
for i in range(len(df3)):
    time = df3["Datetime"][i] 
    df3['Datetime'][i] = time[14:16] # Extract the last two characters of the string

# Creating dummy variables for 15 min interval time
min_dummies = pd.get_dummies(df3["Datetime"]) # One hot encoded the minutes column
df3 = pd.concat([df3, min_dummies], axis=1) # Concatenate the dummy variables with the original DataFrame

# Time series plot of MCP column
df3.MCP.plot() # Plot the MCP column to visualize the trend
plt.show() # Display the plot

# sending the preprocessed data to the database
df3.to_sql('Date_time_MCP_preprocessed_model_based', con = engine, if_exists = 'replace', index = False)
sql = 'SELECT * FROM Date_time_MCP_preprocessed_model_based'# Load the data into a DataFrame called ele_price_forecast
df3_preprocessed = pd.read_sql(sql, con=engine) # Load the data into a DataFrame of date_time_MCP
df3_preprocessed.info() # Display the column names and data types

# Change the directory to the location where you want to save the file
os.chdir("d:\\360 DigiTMG projects\\Project-2\\AI-Powered Electricity Price Forecasting\\8.Datapreprocessing\\Model based Preprocessing")
df3_preprocessed.to_excel("ele_price_preprocessed_model_based.xlsx", index=False) # Save the preprocessed data to an Excel file
df3_preprocessed = df3_preprocessed.drop(columns=["Datetime"]) # Drop the original 'Datetime' column

# Linear fit
linear_fit = np.polyfit(df3_preprocessed["t"], df3_preprocessed["MCP"], 1)
df3_preprocessed["linear_fit"] = np.polyval(linear_fit, df3_preprocessed["t"])

# Exponential fit
exp_fit = np.polyfit(df3_preprocessed["t"], df3_preprocessed["log_MCP"], 1)
df3_preprocessed["exp_fit"] = np.exp(np.polyval(exp_fit, df3_preprocessed["t"]))

# Quadratic fit
quad_fit = np.polyfit(df3_preprocessed["t"], df3_preprocessed["MCP"], 2)
df3_preprocessed["quad_fit"] = np.polyval(quad_fit, df3_preprocessed["t"])

# Plotting the trends
fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
#  plotting for linear trend
axes[0].plot(df3_preprocessed["t"], df3_preprocessed["MCP"], label="MCP", color="blue", alpha=0.6)
axes[0].plot(df3_preprocessed["t"], df3_preprocessed["linear_fit"], label="Linear Trend", color="red")
axes[0].set_title("Linear Trend")
axes[0].legend()
axes[0].grid()
# plotting for exponential trend
axes[1].plot(df3_preprocessed["t"], df3_preprocessed["MCP"], label="MCP", color="blue", alpha=0.6)
axes[1].plot(df3_preprocessed["t"], df3_preprocessed["exp_fit"], label="Exponential Trend", color="green")
axes[1].set_title("Exponential Trend")
axes[1].legend()
axes[1].grid()
# plotting for quadratic trend
axes[2].plot(df3_preprocessed["t"], df3_preprocessed["MCP"], label="MCP", color="blue", alpha=0.6)
axes[2].plot(df3_preprocessed["t"], df3_preprocessed["quad_fit"], label="Quadratic Trend", color="purple")
axes[2].set_title("Quadratic Trend")
axes[2].legend()
axes[2].grid()

plt.xlabel("Time") 
plt.show() # Display the plot


