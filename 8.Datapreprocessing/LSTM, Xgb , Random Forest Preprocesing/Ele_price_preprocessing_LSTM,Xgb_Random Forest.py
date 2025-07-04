
# Importing the required libraries
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical computations
import matplotlib.pyplot as plt  # For plotting graphs
from sqlalchemy import create_engine  # For interacting with databases
from urllib.parse import quote # For encoding passwords that might have special characters
import os  # For file operations
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler # For feature scaling

# Database Credentials
user = 'root' # Database connection credentials
pw = quote('Daya@123') # Encode the password to handle special characters
db = 'ele_price_forecast' # Database name

# Create Database Connection
engine = create_engine(f'mysql+pymysql://{user}:{pw}@localhost/{db}')

# Load Data
sql = 'SELECT * FROM ele_price_forecast_raw;'
df = pd.read_sql(sql, con=engine) # Load the data into a DataFrame of ele_price_forecast_raw
df.info() # Check the information of the DataFrame
df.describe() # Check the summary statistics of the DataFrame


# Remove unwanted columns "in data analysys section MCV is highly correlated with Final Scheduled Volume"
df.drop(columns=["Session ID", 'MCV (MW)' ], inplace=True)

# Rename columns
df.rename(columns={"MCP (Rs/MWh) *": "MCP", 'Purchase Bid (MW)':'Purchase Bid', 'Sell Bid (MW)':'Sell Bid','Final Scheduled Volume (MW)':'Final Scheduled Volume'}, inplace=True)

# Convert 'Datetime' to datetime format
df["Datetime"] = pd.to_datetime(df["Datetime"])

# Identify rows with zero values
zero_rows = df[(df == 0).any(axis=1)]

# Identify zero values and replace with NaN
df.replace(0, np.nan, inplace=True)

# Handle missing values using forward fill and backward fill
df.fillna(method='bfill', inplace=True)

# Checking the null count
df.isnull().sum()
# checking the correlation between the variables
cor = df.corr()

# Sending data to the sql
df.to_sql('ele_price_forecast_preprocessed_Advance', con=engine, if_exists='replace', index=False)

# Load Data from sql
sql = 'SELECT * FROM ele_price_forecast_preprocessed_Advance;'
df = pd.read_sql(sql, con=engine) # Load the data into a DataFrame of 
df.info() # Check the information of the DataFrame
