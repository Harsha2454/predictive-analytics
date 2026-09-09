import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("admission.csv")

print("Dataset Shape:", df.shape)
print(df.head())

df.columns = df.columns.str.strip()

if "Chance of Admit" in df.columns:
    df["Admission"] = (df["Chance of Admit"] >= 0.75).astype(int)
    df = df.drop("Chance of Admit", axis=1)

if "Serial No." in df.columns:
    df = df.drop("Serial No.", axis=1)

X = df.drop("Admission", axis=1)
y = df["Admission"]

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

model = LogisticRegression(random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

results = pd.DataFrame({
    "Actual Admission": y_test.values,
    "Predicted Admission": y_pred
})

print("Prediction Results:")
print(results.head(20))
