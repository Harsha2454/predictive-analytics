import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from imblearn.over_sampling import SMOTE

df = pd.read_csv("imbalanced_data.csv")

print("Dataset Shape:", df.shape)
print(df.head())

print("Missing Values:")
print(df.isnull().sum().sum())

df = df.dropna()

X = df.drop("Target", axis=1)
y = df["Target"]

X = pd.get_dummies(X, drop_first=True)

print("Original Class Distribution:")
print(y.value_counts())

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

model_before = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model_before.fit(X_train_scaled, y_train)

pred_before = model_before.predict(X_test_scaled)

accuracy_before = accuracy_score(
    y_test,
    pred_before
)

f1_before = f1_score(
    y_test,
    pred_before,
    average="macro"
)

print("Before SMOTE")
print("Accuracy:", accuracy_before)
print("Macro F1 Score:", f1_before)

print("Classification Report:")
print(classification_report(y_test, pred_before))

print("Confusion Matrix:")
print(confusion_matrix(y_test, pred_before))

smote = SMOTE(
    random_state=42
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train_scaled,
    y_train
)

print("Class Distribution After SMOTE:")
print(pd.Series(y_train_smote).value_counts())

model_after = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model_after.fit(
    X_train_smote,
    y_train_smote
)

pred_after = model_after.predict(X_test_scaled)

accuracy_after = accuracy_score(
    y_test,
    pred_after
)

f1_after = f1_score(
    y_test,
    pred_after,
    average="macro"
)

print("After SMOTE")
print("Accuracy:", accuracy_after)
print("Macro F1 Score:", f1_after)

print("Classification Report:")
print(classification_report(y_test, pred_after))

print("Confusion Matrix:")
print(confusion_matrix(y_test, pred_after))

comparison = pd.DataFrame({
    "Model": [
        "Before SMOTE",
        "After SMOTE"
    ],
    "Accuracy": [
        accuracy_before,
        accuracy_after
    ],
    "Macro F1 Score": [
        f1_before,
        f1_after
    ]
})

print("Performance Comparison:")
print(comparison)

results = pd.DataFrame({
    "Actual": y_test.values,
    "Before SMOTE": pred_before,
    "After SMOTE": pred_after
})

print("Prediction Results:")
print(results.head(20))
