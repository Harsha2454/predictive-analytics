import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

from sklearn.metrics import mean_absolute_error, mean_squared_error


df = pd.read_csv("energy_consumption.csv")

print("Dataset Shape:", df.shape)
print(df.head())

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date")

df = df.set_index("Date")

df = df[~df.index.duplicated(keep="first")]

print("Missing Values Before Processing:")
print(df["Consumption"].isnull().sum())

df["Consumption"] = df["Consumption"].interpolate(
    method="time"
)

df["Consumption"] = df["Consumption"].fillna(
    method="bfill"
)

df["Consumption"] = df["Consumption"].fillna(
    method="ffill"
)

print("Missing Values After Processing:")
print(df["Consumption"].isnull().sum())

df = df.asfreq("D")

df["Consumption"] = df["Consumption"].interpolate(
    method="time"
)

data = df["Consumption"]

print("Original Mean:", data.mean())
print("Original Standard Deviation:", data.std())

adf_result = adfuller(data)

print("ADF Statistic:", adf_result[0])
print("ADF p-value:", adf_result[1])

if adf_result[1] > 0.05:
    stationary_data = data.diff().dropna()
    print("The series is non-stationary.")
    print("First-order differencing applied.")
else:
    stationary_data = data
    print("The series is stationary.")

adf_stationary = adfuller(stationary_data)

print("Differenced ADF Statistic:", adf_stationary[0])
print("Differenced ADF p-value:", adf_stationary[1])

train_size = int(len(data) * 0.8)

train = data.iloc[:train_size]

test = data.iloc[train_size:]

model = SARIMAX(
    train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7),
    enforce_stationarity=False,
    enforce_invertibility=False
)

model_fit = model.fit(
    disp=False
)

forecast = model_fit.forecast(
    steps=len(test)
)

mae = mean_absolute_error(
    test,
    forecast
)

rmse = np.sqrt(
    mean_squared_error(
        test,
        forecast
    )
)

mape = np.mean(
    np.abs(
        (test - forecast) / test
    )
) * 100

print("Model Evaluation")
print("MAE:", mae)
print("RMSE:", rmse)
print("MAPE:", mape, "%")

results = pd.DataFrame({
    "Actual Consumption": test.values,
    "Predicted Consumption": forecast.values
}, index=test.index)

print("Prediction Results:")
print(results.head(20))

future_model = SARIMAX(
    data,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7),
    enforce_stationarity=False,
    enforce_invertibility=False
)

future_model_fit = future_model.fit(
    disp=False
)

future_forecast = future_model_fit.forecast(
    steps=30
)

future_dates = pd.date_range(
    start=data.index[-1] + pd.Timedelta(days=1),
    periods=30,
    freq="D"
)

future_results = pd.DataFrame({
    "Date": future_dates,
    "Forecasted Consumption": future_forecast.values
})

print("Next 30 Days Energy Forecast:")
print(future_results)

plt.figure(figsize=(14, 6))

plt.plot(
    data.index,
    data.values,
    label="Historical Consumption"
)

plt.xlabel("Date")
plt.ylabel("Energy Consumption")
plt.title("Historical Electricity Consumption")

plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))

plt.plot(
    train.index,
    train.values,
    label="Training Data"
)

plt.plot(
    test.index,
    test.values,
    label="Actual Consumption"
)

plt.plot(
    test.index,
    forecast.values,
    label="Predicted Consumption"
)

plt.xlabel("Date")
plt.ylabel("Energy Consumption")
plt.title("Actual vs Predicted Energy Consumption")

plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))

plt.plot(
    data.index,
    data.values,
    label="Historical Consumption"
)

plt.plot(
    future_dates,
    future_forecast.values,
    label="30-Day Forecast"
)

plt.xlabel("Date")
plt.ylabel("Energy Consumption")
plt.title("Future Electricity Consumption Forecast")

plt.legend()
plt.tight_layout()
plt.show()
