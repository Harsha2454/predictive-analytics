import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from xgboost import XGBClassifier

df = pd.read_csv("imbalanced_multiclass.csv")

print("Dataset Shape:", df.shape)
print(df.head())

print("Missing Values:")
print(df.isnull().sum().sum())

df = df.dropna()

X = df.drop("Target", axis=1)
y = df["Target"]

X = pd.get_dummies(X, drop_first=True)

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

smote = SMOTE(
    random_state=42
)

random_forest = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

rf_pipeline = Pipeline([
    ("scaler", scaler),
    ("smote", smote),
    ("model", random_forest)
])

rf_pipeline.fit(X_train, y_train)

rf_pred = rf_pipeline.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred, average="macro")

print("Random Forest Results")
print("Accuracy:", rf_accuracy)
print("Macro F1 Score:", rf_f1)

print("Classification Report:")
print(classification_report(y_test, rf_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softmax",
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

xgb_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("model", xgb_model)
])

xgb_pipeline.fit(X_train, y_train)

xgb_pred = xgb_pipeline.predict(X_test)

xgb_accuracy = accuracy_score(y_test, xgb_pred)
xgb_f1 = f1_score(y_test, xgb_pred, average="macro")

print("XGBoost Results")
print("Accuracy:", xgb_accuracy)
print("Macro F1 Score:", xgb_f1)

print("Classification Report:")
print(classification_report(y_test, xgb_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, xgb_pred))

ensemble = VotingClassifier(
    estimators=[
        ("random_forest", rf_pipeline),
        ("xgboost", xgb_pipeline)
    ],
    voting="soft"
)

ensemble.fit(X_train, y_train)

ensemble_pred = ensemble.predict(X_test)

ensemble_accuracy = accuracy_score(
    y_test,
    ensemble_pred
)

ensemble_f1 = f1_score(
    y_test,
    ensemble_pred,
    average="macro"
)

print("Voting Ensemble Results")
print("Accuracy:", ensemble_accuracy)
print("Macro F1 Score:", ensemble_f1)

print("Classification Report:")
print(classification_report(y_test, ensemble_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, ensemble_pred))

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

rf_cv_scores = cross_val_score(
    rf_pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring="f1_macro",
    n_jobs=-1
)

xgb_cv_scores = cross_val_score(
    xgb_pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring="f1_macro",
    n_jobs=-1
)

ensemble_cv_scores = cross_val_score(
    ensemble,
    X_train,
    y_train,
    cv=cv,
    scoring="f1_macro",
    n_jobs=-1
)

print("Random Forest Cross-Validation Macro F1:")
print(rf_cv_scores)

print("Random Forest Mean CV Macro F1:")
print(rf_cv_scores.mean())

print("XGBoost Cross-Validation Macro F1:")
print(xgb_cv_scores)

print("XGBoost Mean CV Macro F1:")
print(xgb_cv_scores.mean())

print("Ensemble Cross-Validation Macro F1:")
print(ensemble_cv_scores)

print("Ensemble Mean CV Macro F1:")
print(ensemble_cv_scores.mean())

comparison = pd.DataFrame({
    "Model": [
        "Random Forest",
        "XGBoost",
        "Voting Ensemble"
    ],
    "Test Accuracy": [
        rf_accuracy,
        xgb_accuracy,
        ensemble_accuracy
    ],
    "Test Macro F1": [
        rf_f1,
        xgb_f1,
        ensemble_f1
    ],
    "Mean CV Macro F1": [
        rf_cv_scores.mean(),
        xgb_cv_scores.mean(),
        ensemble_cv_scores.mean()
    ]
})

print("Final Model Comparison:")
print(comparison)

results = pd.DataFrame({
    "Actual Class": y_test.values,
    "Random Forest": rf_pred,
    "XGBoost": xgb_pred,
    "Ensemble": ensemble_pred
})

print("Prediction Results:")
print(results.head(20))
