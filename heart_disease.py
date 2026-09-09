import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("heart_disease.csv")

print("Dataset Shape:", df.shape)
print(df.head())

print("Missing Values:")
print(df.isnull().sum().sum())

df = df.dropna()

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logistic_model.fit(X_train_scaled, y_train)

logistic_pred = logistic_model.predict(X_test_scaled)

logistic_accuracy = accuracy_score(y_test, logistic_pred)

print("Logistic Regression")
print("Accuracy:", logistic_accuracy)

print("Classification Report:")
print(classification_report(y_test, logistic_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, logistic_pred))

forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

forest_model.fit(X_train, y_train)

forest_pred = forest_model.predict(X_test)

forest_accuracy = accuracy_score(y_test, forest_pred)

print("Random Forest Classification")
print("Accuracy:", forest_accuracy)

print("Classification Report:")
print(classification_report(y_test, forest_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, forest_pred))

comparison = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Accuracy": [logistic_accuracy, forest_accuracy]
})

print("Model Comparison:")
print(comparison)

results = pd.DataFrame({
    "Actual": y_test.values,
    "Logistic Prediction": logistic_pred,
    "Random Forest Prediction": forest_pred
})

print("Prediction Results:")
print(results.head(20))
