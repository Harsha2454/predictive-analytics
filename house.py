import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("house_prices.csv")

print("Dataset Shape:", df.shape)
print(df.head())

df = df.dropna()

X = df.drop("price", axis=1)
y = df["price"]

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

ridge_model = Ridge(alpha=1.0)

ridge_model.fit(X_train, y_train)

ridge_pred = ridge_model.predict(X_test)

ridge_mae = mean_absolute_error(y_test, ridge_pred)
ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_pred))
ridge_r2 = r2_score(y_test, ridge_pred)

print("Ridge Regression")
print("MAE:", ridge_mae)
print("RMSE:", ridge_rmse)
print("R2 Score:", ridge_r2)

lasso_model = Lasso(alpha=1.0, max_iter=10000)

lasso_model.fit(X_train, y_train)

lasso_pred = lasso_model.predict(X_test)

lasso_mae = mean_absolute_error(y_test, lasso_pred)
lasso_rmse = np.sqrt(mean_squared_error(y_test, lasso_pred))
lasso_r2 = r2_score(y_test, lasso_pred)

print("Lasso Regression")
print("MAE:", lasso_mae)
print("RMSE:", lasso_rmse)
print("R2 Score:", lasso_r2)

comparison = pd.DataFrame({
    "Model": ["Ridge Regression", "Lasso Regression"],
    "MAE": [ridge_mae, lasso_mae],
    "RMSE": [ridge_rmse, lasso_rmse],
    "R2 Score": [ridge_r2, lasso_r2]
})

print("Model Comparison:")
print(comparison)

results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Ridge Predicted": np.round(ridge_pred, 2),
    "Lasso Predicted": np.round(lasso_pred, 2)
})

print("Prediction Results:")
print(results.head(20))
