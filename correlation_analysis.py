import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("house_prices.csv")

print("Dataset Shape:", df.shape)
print(df.head())

print("Missing Values:")
print(df.isnull().sum().sum())

df = df.dropna()

df = pd.get_dummies(df, drop_first=True)

correlation = df.corr()

print("Correlation Matrix:")
print(correlation)

price_correlation = correlation["price"].sort_values(ascending=False)

print("Correlation with House Price:")
print(price_correlation)

plt.figure(figsize=(12, 8))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Analysis of Factors Affecting House Prices")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))

price_correlation.drop("price").sort_values().plot(
    kind="barh",
    color="steelblue"
)

plt.title("Factors Affecting House Prices")
plt.xlabel("Correlation with Price")
plt.ylabel("Factors")
plt.tight_layout()
plt.show()
