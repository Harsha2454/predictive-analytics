import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv("retail_sales.csv")

print("Dataset Shape:", df.shape)
print(df.head())

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date")

df = df.drop_duplicates("Date")

df = df.set_index("Date")

print("Missing Values:")
print(df["Sales"].isnull().sum())

df["Sales"] = df["Sales"].interpolate(
    method="time"
)

df["Sales"] = df["Sales"].ffill()

df["Sales"] = df["Sales"].bfill()

print("Sales Statistics:")
print(df["Sales"].describe())

sales = df["Sales"]

train_size = int(len(sales) * 0.8)

train = sales.iloc[:train_size]

test = sales.iloc[train_size:]

model = ARIMA(
    train,
    order=(2, 1, 2)
)

model_fit = model.fit()

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

print("ARIMA Model Evaluation")
print("MAE:", mae)
print("RMSE:", rmse)
print("MAPE:", mape, "%")

results = pd.DataFrame({
    "Actual Sales": test.values,
    "Predicted Sales": forecast.values
}, index=test.index)

print("Forecast Results:")
print(results.head(20))

future_model = ARIMA(
    sales,
    order=(2, 1, 2)
)

future_model_fit = future_model.fit()

future_periods = 12

future_forecast = future_model_fit.forecast(
    steps=future_periods
)

last_date = sales.index[-1]

date_difference = sales.index.to_series().diff().median()

if date_difference <= pd.Timedelta(days=8):
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(weeks=1),
        periods=future_periods,
        freq="W"
    )
else:
    future_dates = pd.date_range(
        start=last_date + pd.offsets.MonthBegin(1),
        periods=future_periods,
        freq="MS"
    )

future_results = pd.DataFrame({
    "Date": future_dates,
    "Forecasted Sales": np.round(
        future_forecast.values,
        2
    )
})

print("Future Sales Forecast:")
print(future_results)

plt.figure(figsize=(14, 6))

plt.plot(
    sales.index,
    sales.values,
    label="Historical Sales"
)

plt.plot(
    test.index,
    forecast.values,
    label="Predicted Sales"
)

plt.xlabel("Date")
plt.ylabel("Sales")

plt.title("Actual vs Predicted Retail Sales")

plt.legend()

plt.tight_layout()

plt.show()

plt.figure(figsize=(14, 6))

plt.plot(
    sales.index,
    sales.values,
    label="Historical Sales"
)

plt.plot(
    future_dates,
    future_forecast.values,
    label="Future Forecast"
)

plt.xlabel("Date")
plt.ylabel("Sales")

plt.title("Future Retail Sales Forecast")

plt.legend()

plt.tight_layout()

plt.show()

plt.figure(figsize=(10, 5))

plt.bar(
    future_dates.astype(str),
    future_forecast.values,
    color="steelblue"
)

plt.xlabel("Future Period")

plt.ylabel("Forecasted Sales")

plt.title("Forecasted Retail Sales")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()
