import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


df = pd.read_csv("taxi_demand.csv")

print("Dataset Shape:", df.shape)
print(df.head())

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date")

df = df.dropna()

print("Missing Values:")
print(df.isnull().sum().sum())

df["Hour"] = df["Date"].dt.hour
df["Day"] = df["Date"].dt.day
df["DayOfWeek"] = df["Date"].dt.dayofweek
df["Month"] = df["Date"].dt.month
df["Year"] = df["Date"].dt.year
df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)

df["IsWeekend"] = (
    df["DayOfWeek"] >= 5
).astype(int)

df["Lag_1"] = df["Demand"].shift(1)
df["Lag_24"] = df["Demand"].shift(24)

df["Rolling_24"] = (
    df["Demand"]
    .rolling(24)
    .mean()
)

df = df.dropna()

print("Average Taxi Demand:", df["Demand"].mean())
print("Maximum Taxi Demand:", df["Demand"].max())
print("Minimum Taxi Demand:", df["Demand"].min())

hourly_demand = df.groupby("Hour")["Demand"].mean()

print("Average Demand by Hour:")
print(hourly_demand)

daily_demand = df.groupby("DayOfWeek")["Demand"].mean()

print("Average Demand by Day:")
print(daily_demand)

monthly_demand = df.groupby("Month")["Demand"].mean()

print("Average Demand by Month:")
print(monthly_demand)

features = [
    "Hour",
    "Day",
    "DayOfWeek",
    "Month",
    "Year",
    "WeekOfYear",
    "IsWeekend",
    "Lag_1",
    "Lag_24",
    "Rolling_24"
]

X = df[features]

y = df["Demand"]

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]

X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]

y_test = y.iloc[split_index:]

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

prediction = model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    prediction
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        prediction
    )
)

print("Model Evaluation")
print("MAE:", mae)
print("RMSE:", rmse)

results = pd.DataFrame({
    "Date": df["Date"].iloc[split_index:].values,
    "Actual Demand": y_test.values,
    "Predicted Demand": np.round(prediction, 2)
})

print("Prediction Results:")
print(results.head(20))

results["Required Taxis"] = np.ceil(
    results["Predicted Demand"]
).astype(int)

print("Fleet Utilisation Results:")
print(
    results[
        [
            "Date",
            "Predicted Demand",
            "Required Taxis"
        ]
    ].head(20)
)

peak_hour = hourly_demand.idxmax()

peak_demand = hourly_demand.max()

low_hour = hourly_demand.idxmin()

low_demand = hourly_demand.min()

print("Peak Demand Hour:", peak_hour)
print("Peak Average Demand:", peak_demand)

print("Lowest Demand Hour:", low_hour)
print("Lowest Average Demand:", low_demand)

feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    "Importance",
    ascending=False
)

print("Feature Importance:")
print(feature_importance)

plt.figure(figsize=(12, 6))

plt.plot(
    df["Date"],
    df["Demand"],
    label="Taxi Demand"
)

plt.xlabel("Date")
plt.ylabel("Number of Rides")
plt.title("Taxi Ride Demand Over Time")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))

plt.plot(
    hourly_demand.index,
    hourly_demand.values,
    marker="o"
)

plt.xlabel("Hour of Day")
plt.ylabel("Average Taxi Demand")
plt.title("Taxi Demand Seasonality by Hour")
plt.xticks(range(24))
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))

plt.bar(
    daily_demand.index,
    daily_demand.values
)

plt.xlabel("Day of Week")
plt.ylabel("Average Taxi Demand")
plt.title("Taxi Demand by Day of Week")
plt.xticks(
    range(7),
    [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]
)

plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))

plt.plot(
    y_test.index,
    y_test.values,
    label="Actual Demand"
)

plt.plot(
    y_test.index,
    prediction,
    label="Predicted Demand"
)

plt.xlabel("Time")
plt.ylabel("Taxi Demand")
plt.title("Actual vs Predicted Taxi Demand")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))

plt.bar(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Taxi Demand Prediction Feature Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
