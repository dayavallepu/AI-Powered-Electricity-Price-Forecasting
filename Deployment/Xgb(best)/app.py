# Import necessary libraries
import pandas as pd  # for data manipulation
import streamlit as st  # for creating web apps
import numpy as np  # for numerical calculations
import joblib  # for saving and loading models
import plotly.graph_objects as go  # for interactive plots
from sqlalchemy import create_engine  # for interacting with databases
from urllib.parse import quote  # for URL encoding
from PIL import Image  # for image processing
import os  # for operating system interactions
import pymysql  # for MySQL database interactions
import xgboost as xgb  # for XGBoost model

# Set working directory
os.chdir(r"D:\360 DigiTMG projects\Project-2\AI-Powered Electricity Price Forecasting\10. Deployment\Xgb(best)")

# Load the trained XGBoost model and scaler
model = joblib.load("final_xgboost_model.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit App Layout
st.set_page_config(page_title="MCP Forecasting", layout="wide")

def preprocess_data(df):
    """Preprocess user-uploaded data for forecasting."""
    df["Datetime"] = pd.to_datetime(df["Datetime"])  # Convert 'Datetime' column to datetime format
    df.drop(columns=["Session ID", "MCV (MW)", "Unnamed: 0"], errors="ignore", inplace=True)  # Drop unwanted columns
    df.rename(columns={"MCP (Rs/MWh) *": "MCP"}, inplace=True)  # Rename columns
    df.replace(0, np.nan, inplace=True)  # Replace zero values with NaN
    df.fillna(method="bfill", inplace=True)  # Fill NaN values using backward fill
    
    # Create lag features
    for i in range(1, 97):
        df[f'MCP_lag_{i}'] = df['MCP'].shift(i)
    
    df.dropna(inplace=True)  # Drop rows with NaN values
    
    # Prepare feature matrix
    X = df.drop(columns=["MCP", "Datetime"])
    X_scaled = scaler.transform(X)  # Scale features
    return df, X_scaled

def forecast_next_steps(X_scaled, df, steps, confidence_level):
    """Forecast MCP values for the next 'steps' time periods."""
    last_known_lags = X_scaled[-1].copy()  # Get the last known lag features
    future_predictions, upper_bound, lower_bound = [], [], []  # Initialize lists for predictions and confidence intervals

    y_actual = df["MCP"].values  # Actual MCP values
    y_pred = model.predict(X_scaled)  # Predicted MCP values
    error_std = np.std(y_actual - y_pred)  # Standard deviation of errors

    # Confidence interval mapping
    confidence_mapping = {95: 1.96, 90: 1.645, 99: 2.576}
    confidence_multiplier = confidence_mapping[confidence_level]
    confidence_interval = confidence_multiplier * error_std  # Calculate confidence interval

    for _ in range(steps):
        next_pred = model.predict(last_known_lags.reshape(1, -1))[0]  # Predict the next MCP

        # Compute confidence intervals
        upper_ci = next_pred + confidence_interval
        lower_ci = next_pred - confidence_interval

        # Store predictions
        future_predictions.append(next_pred)
        upper_bound.append(upper_ci)
        lower_bound.append(lower_ci)

        # Update lag features properly
        next_pred_scaled = scaler.transform(np.array([next_pred] * X_scaled.shape[1]).reshape(1, -1))
        last_known_lags = np.roll(last_known_lags, -1)  # Shift window
        last_known_lags[-1] = next_pred_scaled[0, 0]  # Add new scaled prediction

    return future_predictions, upper_bound, lower_bound

def retrain_model(new_data):
    """Retrains the XGBoost model with new data."""
    new_data, X_new_scaled = preprocess_data(new_data)  # Preprocess new data
    y_new = new_data["MCP"].values  # Extract target values
    X_new = X_new_scaled  # Extract features

    # Define new model with updated hyperparameters
    new_model = xgb.XGBRegressor(
        objective='reg:squarederror', 
        n_estimators=200,  # Increase estimators for better learning
        learning_rate=0.03,  # Reduce learning rate for gradual improvements
        max_depth=8,  # Increase depth to capture more patterns
        subsample=0.8,  # Avoid overfitting by using 80% of data per tree
        colsample_bytree=0.8,
        reg_lambda=0.5,  # Reduce L2 regularization
        reg_alpha=0.1  # Reduce L1 regularization
    )

    new_model.fit(X_new, y_new)  # Fit the new model
    
    joblib.dump(new_model, "final_xgboost_model.pkl")  # Save the new model
    st.success(f"✅ Model retrained and updated successfully!\nStart_date: {st.session_state.training_start_date}\nEnd_date: {st.session_state.training_end_date}")

def main():
    """Streamlit application for MCP forecasting."""
    
    # Sidebar Logo
    image = Image.open("AiSPRY logo.jpg")
    st.sidebar.image(image)

    # Title Styling
    st.markdown(
        """
        <style>
        .title-box {
            background-color: #FF5733;
            color: white;
            text-align: center;
            padding: 15px;
            border-radius: 15px;
            font-size: 24px;
            font-weight: bold;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.2);
        }
        .button {
            background-color: orange;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            cursor: pointer;
            box-shadow: 3px 3px 8px rgba(0, 0, 0, 0.3);
            width: 150px;  /* Resized button */
        }
        .button:hover {
            background-color: darkorange;
        }
        .center {
            display: flex;
            justify-content: center;
        }
        </style>
        <div class="title-box">  Electricity Price Forecasting  </div>
        """, unsafe_allow_html=True
    )

    # Sidebar Inputs
    user = st.sidebar.text_input("User", "Type Here")  # User input for database username
    pw = st.sidebar.text_input("Password", "Type Here", type="password")  # User input for database password
    db = st.sidebar.text_input("Database", "Type Here")  # User input for database name
    forecast_steps = st.sidebar.number_input("📊 Forecast Steps", min_value=1, max_value=500, value=200, step=1)  # Number of forecast steps
    confidence_level = st.sidebar.selectbox("✅ Confidence Interval", [95, 90, 99], index=0)  # Confidence interval selection

    # File Uploader
    uploaded_file = st.sidebar.file_uploader("📂 Upload CSV or Excel For Forecasting", type=["csv", "xlsx"])  # File uploader for new data

    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith("csv") else pd.read_excel(uploaded_file)  # Load uploaded file
        df, X_scaled = preprocess_data(df)  # Preprocess the data
        # Update training start & end dates based on actual MCP values in uploaded file
        st.session_state["training_start_date"] = df["Datetime"].min().strftime("%Y-%m-%d %H:%M")
        st.session_state["training_end_date"] = df["Datetime"].max().strftime("%Y-%m-%d %H:%M")
        # Sidebar Forecasting Info Box
        st.sidebar.markdown(f"""
        <div style='padding: 10px; background-color: lightgreen; border-radius: 10px;'>
            <h3 style='text-align: center;'>Training Data Info</h3>
            <p><strong>Start Date:</strong> {st.session_state['training_start_date']}</p>
            <p><strong>End Date:</strong> {st.session_state['training_end_date']}</p>
        </div>
        """, unsafe_allow_html=True)
        # Predict Button (Aligned to the Left)
        st.markdown(
            """
            <style>
            .stButton button {
                background-color: orange;
                color: white;
                font-size: 16px;
                width: 150px;
                border-radius: 10px;
                border: none;
                box-shadow: 3px 3px 8px rgba(0, 0, 0, 0.3);
                margin-left: 0;  /* Align to the left */
            }
            .stButton button:hover {
                background-color: darkorange;
            }
            </style>
            """, unsafe_allow_html=True
        )
        predict_clicked = st.button("Predict", key="predict", help="Click to generate forecasts")  # Predict button

        if predict_clicked:
            future_pred, upper_ci, lower_ci = forecast_next_steps(X_scaled, df, forecast_steps, confidence_level)  # Generate forecasts
            future_dates = pd.date_range(start=df["Datetime"].iloc[-1], periods=len(future_pred) + 1, freq="15T")[1:]  # Generate future dates

            forecast_df = pd.DataFrame({
                "Datetime": future_dates,
                "Predicted_MCP": future_pred,
                "Upper_CI": upper_ci,
                "Lower_CI": lower_ci
            })
            st.session_state["forecast_df"] = forecast_df  # Store forecast data in session state
            # Show Plot First
             # Get start and end dates for forecast
            start_date = forecast_df["Datetime"].min().strftime("%Y-%m-%d %H:%M")
            end_date = forecast_df["Datetime"].max().strftime("%Y-%m-%d %H:%M")
            st.write(f"### 📈 MCP Forecasted Plot from {start_date}  to  {end_date}")
            fig = go.Figure()
            
            # Forecasted MCP in darkblue
            fig.add_trace(go.Scatter(x=forecast_df["Datetime"], y=forecast_df["Predicted_MCP"],
                         mode="lines+markers", name="Forecasted MCP", line=dict(color="darkblue")))

            # Upper and Lower CI in Light Sky Blue
            fig.add_trace(go.Scatter(x=forecast_df["Datetime"], y=forecast_df["Upper_CI"],
                                    fill=None, mode="lines", name="Upper CI", line=dict(dash="dot", color="lightskyblue")))
            fig.add_trace(go.Scatter(x=forecast_df["Datetime"], y=forecast_df["Lower_CI"],
                                    fill="tonexty", mode="lines", name="Lower CI", line=dict(dash="dot", color="lightskyblue")))

            fig.update_layout(title="MCP Forecast", xaxis_title="Datetime", yaxis_title="MCP",
                            xaxis_rangeslider_visible=True)
            st.plotly_chart(fig)  # Display plot

            # Show Table Below the Plot
            st.write(f"### 📊 Forecasted Data from  {start_date}  to  {end_date}")
            st.dataframe(forecast_df)  # Display forecast data

    if "forecast_df" in st.session_state:   # Save to Database Button (With Shadow)
        st.markdown("<div class='center'>", unsafe_allow_html=True)
        save_clicked = st.button("💾 Save to Database", key="save", help="Click to store predictions", use_container_width=True)  # Save to Database button
        st.markdown("</div>", unsafe_allow_html=True)

        if save_clicked and user and pw and db:                
            try:
                engine = create_engine(f"mysql+pymysql://{user}:{quote(pw)}@localhost/{db}")  # Create database connection
                st.session_state["forecast_df"].to_sql("mcp_forecast_results", con=engine, if_exists="replace", index=False)  # Save forecast data to database
                st.success("✅ Results successfully stored in the database.")  # Success message
            except Exception as e:
                st.error(f"❌ Error saving to database: {e}")  # Error message
    

    uploaded_train_file = st.sidebar.file_uploader("📂 Upload New Training Data", type=["csv", "xlsx"])  # File uploader for new training data
    if uploaded_train_file and st.sidebar.button("🔄 Retrain Model"):  # Retrain model button
        new_train_data = pd.read_csv(uploaded_train_file) if uploaded_train_file.name.endswith("csv") else pd.read_excel(uploaded_train_file)  # Load new training data
        retrain_model(new_train_data)  # Retrain the model

if __name__ == "__main__":
    main()  # Run the main function
