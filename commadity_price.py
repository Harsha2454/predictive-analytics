import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv("commodity_prices.csv")

print("Dataset Shape:", df.shape)
print(df.head())

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date")

df = df.drop_duplicates("Date")

df = df.set_index("Date")

print("Missing Values:")
print(df.isnull().sum())

df = df.interpolate(method="time")

df = df.ffill()
df = df.bfill()

df["Return"] = df["Close"].pct_change()

df["MA_7"] = df["Close"].rolling(7).mean()

df["MA_30"] = df["Close"].rolling(30).mean()

df["Volatility"] = df["Return"].rolling(7).std()

df["Price_Change"] = df["Close"].diff()

df = df.dropna()

print("Price Statistics:")
print(df["Close"].describe())

correlation = df.corr(numeric_only=True)

print("Correlation Matrix:")
print(correlation)

print("Correlation with Commodity Price:")
print(
    correlation["Close"]
    .sort_values(ascending=False)
)

features = [
    "Open",
    "High",
    "Low",
    "Volume",
    "Return",
    "MA_7",
    "MA_30",
    "Volatility"
]

X = df[features]

y = df["Close"]

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]

X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]

y_test = y.iloc[split_index:]

model = SARIMAX(
    y_train,
    exog=X_train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7),
    enforce_stationarity=False,
    enforce_invertibility=False
)

model_fit = model.fit(
    disp=False
)

forecast = model_fit.forecast(
    steps=len(X_test),
    exog=X_test
)

mae = mean_absolute_error(
    y_test,
    forecast
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        forecast
    )
)

mape = np.mean(
    np.abs(
        (y_test - forecast) / y_test
    )
) * 100

print("Model Evaluation")
print("MAE:", mae)
print("RMSE:", rmse)
print("MAPE:", mape, "%")

results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": forecast.values
}, index=y_test.index)

print("Forecast Results:")
print(results.head(20))

future_model = SARIMAX(
    y,
    exog=X,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7),
    enforce_stationarity=False,
    enforce_invertibility=False
)

future_model_fit = future_model.fit(
    disp=False
)

future_forecast = future_model_fit.forecast(
    steps=30,
    exog=X.iloc[-30:]
)

future_dates = pd.date_range(
    start=df.index[-1] + pd.Timedelta(days=1),
    periods=30,
    freq="D"
)

future_results = pd.DataFrame({
    "Date": future_dates,
    "Forecasted Price": future_forecast.values
})

print("Next 30 Days Commodity Price Forecast:")
print(future_results)

plt.figure(figsize=(14, 6))

plt.plot(
    df.index,
    df["Close"],
    label="Commodity Price"
)

plt.plot(
    df.index,
    df["MA_7"],
    label="7-Day Moving Average"
)

plt.plot(
    df.index,
    df["MA_30"],
    label="30-Day Moving Average"
)

plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Commodity Price Movement")

plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))

plt.plot(
    y_test.index,
    y_test.values,
    label="Actual Price"
)

plt.plot(
    y_test.index,
    forecast.values,
    label="Predicted Price"
)

plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Actual vs Predicted Commodity Prices")

plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))

plt.plot(
    df.index,
    df["Return"],
    color="purple"
)

plt.xlabel("Date")
plt.ylabel("Daily Return")

plt.title("Commodity Market Fluctuations")

plt.axhline(
    0,
    color="black",
    linestyle="--"
)

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))

correlation["Close"].sort_values().plot(
    kind="barh",
    color="steelblue"
)

plt.xlabel("Correlation")
plt.ylabel("Factors")

plt.title("Factors Driving Commodity Price Movements")

plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))

plt.plot(
    df.index,
    df["Close"],
    label="Historical Price"
)

plt.plot(
    future_dates,
    future_forecast.values,
    label="30-Day Forecast"
)

plt.xlabel("Date")
plt.ylabel("Price")

plt.title("Future Commodity Price Forecast")

plt.legend()
plt.tight_layout()
plt.show()
