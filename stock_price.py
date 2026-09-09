import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df = pd.read_csv("stock_ohlc.csv")

print("Dataset Shape:", df.shape)
print(df.head())

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date")

df = df.dropna()

df = df.set_index("Date")

print("Missing Values:")
print(df.isnull().sum().sum())

df["Daily_Return"] = df["Close"].pct_change()

df["High_Low_Range"] = (
    df["High"] - df["Low"]
) / df["Close"]

df["Open_Close_Range"] = (
    df["Close"] - df["Open"]
) / df["Open"]

df["MA_5"] = df["Close"].rolling(5).mean()

df["MA_10"] = df["Close"].rolling(10).mean()

df["MA_20"] = df["Close"].rolling(20).mean()

df["Volatility"] = df["Daily_Return"].rolling(10).std()

df["Target"] = (
    df["Close"].shift(-1) > df["Close"]
).astype(int)

df = df.dropna()

features = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Daily_Return",
    "High_Low_Range",
    "Open_Close_Range",
    "MA_5",
    "MA_10",
    "MA_20",
    "Volatility"
]

X = df[features]

y = df["Target"]

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]

X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]

y_test = y.iloc[split_index:]

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(
    X_train,
    y_train
)

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("Model Accuracy:", accuracy)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

results = df.iloc[split_index:].copy()

results["Prediction"] = y_pred

results["Strategy_Return"] = (
    results["Prediction"] * results["Daily_Return"]
)

results["Market_Return"] = results["Daily_Return"]

results["Strategy_Cumulative"] = (
    1 + results["Strategy_Return"]
).cumprod()

results["Market_Cumulative"] = (
    1 + results["Market_Return"]
).cumprod()

strategy_return = (
    results["Strategy_Cumulative"].iloc[-1] - 1
) * 100

market_return = (
    results["Market_Cumulative"].iloc[-1] - 1
) * 100

print("Trading Strategy Return:", strategy_return, "%")

print("Buy and Hold Return:", market_return, "%")

initial_value = 100000

final_strategy_value = (
    initial_value *
    results["Strategy_Cumulative"].iloc[-1]
)

final_market_value = (
    initial_value *
    results["Market_Cumulative"].iloc[-1]
)

print(
    "Initial Investment:",
    initial_value
)

print(
    "Final Strategy Value:",
    final_strategy_value
)

print(
    "Final Buy and Hold Value:",
    final_market_value
)

results["Strategy_Price"] = (
    results["Strategy_Cumulative"] *
    results["Close"].iloc[0]
)

results["Market_Price"] = (
    results["Market_Cumulative"] *
    results["Close"].iloc[0]
)

plt.figure(figsize=(14, 6))

plt.plot(
    df.index,
    df["Close"],
    label="Closing Price"
)

plt.plot(
    df.index,
    df["MA_20"],
    label="20-Day Moving Average"
)

plt.xlabel("Date")

plt.ylabel("Price")

plt.title("OHLC Stock Price Analysis")

plt.legend()

plt.tight_layout()

plt.show()

plt.figure(figsize=(14, 6))

plt.plot(
    results.index,
    results["Market_Cumulative"],
    label="Buy and Hold"
)

plt.plot(
    results.index,
    results["Strategy_Cumulative"],
    label="ML Trading Strategy"
)

plt.xlabel("Date")

plt.ylabel("Portfolio Growth")

plt.title("Trading Strategy Performance")

plt.legend()

plt.tight_layout()

plt.show()

plt.figure(figsize=(14, 6))

plt.plot(
    results.index,
    results["Close"],
    label="Actual Price"
)

buy_signals = results[
    results["Prediction"] == 1
]

sell_signals = results[
    results["Prediction"] == 0
]

plt.scatter(
    buy_signals.index,
    buy_signals["Close"],
    marker="^",
    color="green",
    label="Buy Signal"
)

plt.scatter(
    sell_signals.index,
    sell_signals["Close"],
    marker="v",
    color="red",
    label="Sell Signal"
)

plt.xlabel("Date")

plt.ylabel("Price")

plt.title("ML-Based Trading Signals")

plt.legend()

plt.tight_layout()

plt.show()

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
