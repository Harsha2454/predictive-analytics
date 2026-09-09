import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler

df = pd.read_csv("customer_data.csv")

print("Dataset Shape:", df.shape)
print(df.head())

print("Missing Values:")
print(df.isnull().sum().sum())

df = df.dropna()

X = df.drop("Churn", axis=1)
y = df["Churn"]

if "customerID" in X.columns:
    X = X.drop("customerID", axis=1)

y = y.map({
    "Yes": 1,
    "No": 0
})

X = pd.get_dummies(X, drop_first=True)

print("Class Distribution Before Sampling:")
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model_before = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model_before.fit(X_train, y_train)

pred_before = model_before.predict(X_test)

accuracy_before = accuracy_score(y_test, pred_before)

print("Before Sampling")
print("Accuracy:", accuracy_before)

print("Classification Report:")
print(classification_report(y_test, pred_before))

print("Confusion Matrix:")
print(confusion_matrix(y_test, pred_before))

sampler = RandomOverSampler(
    random_state=42
)

X_train_resampled, y_train_resampled = sampler.fit_resample(
    X_train,
    y_train
)

print("Class Distribution After Sampling:")
print(y_train_resampled.value_counts())

model_after = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model_after.fit(X_train_resampled, y_train_resampled)

pred_after = model_after.predict(X_test)

accuracy_after = accuracy_score(y_test, pred_after)

print("After Random Oversampling")
print("Accuracy:", accuracy_after)

print("Classification Report:")
print(classification_report(y_test, pred_after))

print("Confusion Matrix:")
print(confusion_matrix(y_test, pred_after))

comparison = pd.DataFrame({
    "Model": [
        "Before Sampling",
        "After Random Oversampling"
    ],
    "Accuracy": [
        accuracy_before,
        accuracy_after
    ]
})

print("Model Comparison:")
print(comparison)

results = pd.DataFrame({
    "Actual": y_test.values,
    "Before Sampling": pred_before,
    "After Sampling": pred_after
})

print("Prediction Results:")
print(results.head(20))
