import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv("sales_data.csv")

print("Dataset Shape:", df.shape)
print(df.head())

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date")

df = df.dropna()

df = df.set_index("Date")

print("Missing Values:")
print(df.isnull().sum().sum())

sales = df["Sales"]

train_size = int(len(sales) * 0.8)

train = sales[:train_size]
test = sales[train_size:]

model = ARIMA(
    train,
    order=(5, 1, 0)
)

model_fit = model.fit()

forecast = model_fit.forecast(
    steps=len(test)
)

mae = mean_absolute_error(test, forecast)
rmse = np.sqrt(mean_squared_error(test, forecast))

print("ARIMA Model Results")
print("MAE:", mae)
print("RMSE:", rmse)

results = pd.DataFrame({
    "Actual Sales": test.values,
    "Predicted Sales": np.round(forecast.values, 2)
})

print("Forecast Results:")
print(results.head(20))

future_model = ARIMA(
    sales,
    order=(5, 1, 0)
)

future_model_fit = future_model.fit()

future_forecast = future_model_fit.forecast(
    steps=30
)

future_dates = pd.date_range(
    start=sales.index[-1] + pd.Timedelta(days=1),
    periods=30,
    freq="D"
)

future_results = pd.DataFrame({
    "Date": future_dates,
    "Forecasted Sales": np.round(future_forecast.values, 2)
})

print("Next 30 Days Sales Forecast:")
print(future_results)

plt.figure(figsize=(12, 6))

plt.plot(
    train.index,
    train.values,
    label="Training Sales"
)

plt.plot(
    test.index,
    test.values,
    label="Actual Sales"
)

plt.plot(
    test.index,
    forecast.values,
    label="Predicted Sales"
)

plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Sales Forecasting Using ARIMA")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))

plt.plot(
    sales.index,
    sales.values,
    label="Historical Sales"
)

plt.plot(
    future_dates,
    future_forecast.values,
    label="30-Day Forecast"
)

plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Future Sales Forecast")
plt.legend()
plt.tight_layout()
plt.show()
