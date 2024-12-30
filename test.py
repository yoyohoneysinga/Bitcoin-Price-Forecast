import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

# Load the data
data = pd.read_csv("Bitcoin Price Prediction/data.csv", index_col="Date", parse_dates=True)

# Fill missing values using forward fill
data.fillna(method='ffill', inplace=True)

# Use 'Close' for forecasting
y = data['Close']

# Split the data into train and test sets
train_size = int(len(y) * 0.8)
train, test = y[:train_size], y[train_size:]

# Function to compute MAE and RMSE
def compute_errors(true, forecast):
    mae = mean_absolute_error(true, forecast)
    rmse = sqrt(mean_squared_error(true, forecast))
    return mae, rmse

# ARIMA Model
arima_model = auto_arima(train, seasonal=False, m=1, trace=True, suppress_warnings=True)
arima_forecast = arima_model.predict(n_periods=len(test))
arima_mae, arima_rmse = compute_errors(test, arima_forecast)

# SARIMA Model
sarima_model = SARIMAX(train, order=(5,1,0), seasonal_order=(1,1,0,7))  # (p,d,q) (P,D,Q,s)
sarima_fitted = sarima_model.fit()
sarima_forecast = sarima_fitted.forecast(steps=len(test))
sarima_mae, sarima_rmse = compute_errors(test, sarima_forecast)

# Holt-Winters Exponential Smoothing Model
holt_winters_model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=7)
holt_winters_fitted = holt_winters_model.fit()
holt_winters_forecast = holt_winters_fitted.forecast(steps=len(test))
holt_winters_mae, holt_winters_rmse = compute_errors(test, holt_winters_forecast)

# Simple Exponential Smoothing Model
ses_model = ExponentialSmoothing(train, trend=None, seasonal=None)
ses_fitted = ses_model.fit()
ses_forecast = ses_fitted.forecast(steps=len(test))
ses_mae, ses_rmse = compute_errors(test, ses_forecast)

# Create a comparison table of the errors
comparison_table = pd.DataFrame({
    'Model': ['ARIMA', 'SARIMA', 'Holt-Winters', 'Simple Exponential Smoothing'],
    'MAE': [arima_mae, sarima_mae, holt_winters_mae, ses_mae],
    'RMSE': [arima_rmse, sarima_rmse, holt_winters_rmse, ses_rmse]
})

print(comparison_table)

# Visualize the forecasts
plt.figure(figsize=(12,6))
plt.plot(train.index, train, label='Train Data')
plt.plot(test.index, test, label='Test Data', color='black')
plt.plot(test.index, arima_forecast, label='ARIMA Forecast', color='blue')
plt.plot(test.index, sarima_forecast, label='SARIMA Forecast', color='green')
plt.plot(test.index, holt_winters_forecast, label='Holt-Winters Forecast', color='red')
plt.plot(test.index, ses_forecast, label='SES Forecast', color='orange')
plt.title('Bitcoin Price Forecast Comparison')
plt.legend()
plt.show()
