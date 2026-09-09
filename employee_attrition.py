import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("employee_attrition.csv")

print("Dataset Shape:", df.shape)
print(df.head())

print("Missing Values:")
print(df.isnull().sum().sum())

df = df.dropna()

df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

X = df.drop("Attrition", axis=1)
y = df["Attrition"]

if "EmployeeNumber" in X.columns:
    X = X.drop("EmployeeNumber", axis=1)

if "EmployeeCount" in X.columns:
    X = X.drop("EmployeeCount", axis=1)

if "StandardHours" in X.columns:
    X = X.drop("StandardHours", axis=1)

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

results = pd.DataFrame({
    "Actual Attrition": y_test.values,
    "Predicted Attrition": y_pred
})

print("Prediction Results:")
print(results.head(20))

print("Employees Predicted to Leave:", (y_pred == 1).sum())
print("Employees Predicted to Stay:", (y_pred == 0).sum())
