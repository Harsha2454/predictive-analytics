import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("creditcard.csv")

print("Dataset Shape:", df.shape)
print(df.head())

print("Missing Values:")
print(df.isnull().sum().sum())

df = df.dropna()

X = df.drop("Class", axis=1)
y = df["Class"]

print("Class Distribution:")
print(y.value_counts())

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

normal_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

normal_model.fit(X_train, y_train)

normal_pred = normal_model.predict(X_test)

normal_accuracy = accuracy_score(y_test, normal_pred)

print("Normal Logistic Regression")
print("Accuracy:", normal_accuracy)

print("Classification Report:")
print(classification_report(y_test, normal_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, normal_pred))

cost_sensitive_model = LogisticRegression(
    class_weight={0: 1, 1: 10},
    max_iter=1000,
    random_state=42
)

cost_sensitive_model.fit(X_train, y_train)

cost_sensitive_pred = cost_sensitive_model.predict(X_test)

cost_sensitive_accuracy = accuracy_score(
    y_test,
    cost_sensitive_pred
)

print("Cost-Sensitive Logistic Regression")
print("Accuracy:", cost_sensitive_accuracy)

print("Classification Report:")
print(classification_report(y_test, cost_sensitive_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, cost_sensitive_pred))

comparison = pd.DataFrame({
    "Model": [
        "Normal Logistic Regression",
        "Cost-Sensitive Logistic Regression"
    ],
    "Accuracy": [
        normal_accuracy,
        cost_sensitive_accuracy
    ]
})

print("Model Comparison:")
print(comparison)

results = pd.DataFrame({
    "Actual": y_test.values,
    "Normal Prediction": normal_pred,
    "Cost-Sensitive Prediction": cost_sensitive_pred
})

print("Prediction Results:")
print(results.head(20))

print(
    "Fraud Detected by Normal Model:",
    (normal_pred == 1).sum()
)

print(
    "Fraud Detected by Cost-Sensitive Model:",
    (cost_sensitive_pred == 1).sum()
)
