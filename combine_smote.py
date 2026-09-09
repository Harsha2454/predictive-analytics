import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline

from xgboost import XGBClassifier

df = pd.read_csv("fraud_intrusion_data.csv")

print("Dataset Shape:", df.shape)
print(df.head())

print("Missing Values:")
print(df.isnull().sum().sum())

df = df.dropna()

target = "Class"

X = df.drop(target, axis=1)
y = df[target]

X = pd.get_dummies(X, drop_first=True)

print("Number of Features:", X.shape[1])

print("Original Class Distribution:")
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

smote_rf = Pipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("model", RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    ))
])

smote_rf.fit(X_train, y_train)

smote_rf_pred = smote_rf.predict(X_test)

smote_rf_accuracy = accuracy_score(
    y_test,
    smote_rf_pred
)

smote_rf_f1 = f1_score(
    y_test,
    smote_rf_pred,
    average="macro"
)

print("SMOTE + Random Forest")
print("Accuracy:", smote_rf_accuracy)
print("Macro F1 Score:", smote_rf_f1)

print("Classification Report:")
print(classification_report(y_test, smote_rf_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, smote_rf_pred))

adasyn_rf = Pipeline([
    ("scaler", StandardScaler()),
    ("adasyn", ADASYN(random_state=42)),
    ("model", RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    ))
])

adasyn_rf.fit(X_train, y_train)

adasyn_rf_pred = adasyn_rf.predict(X_test)

adasyn_rf_accuracy = accuracy_score(
    y_test,
    adasyn_rf_pred
)

adasyn_rf_f1 = f1_score(
    y_test,
    adasyn_rf_pred,
    average="macro"
)

print("ADASYN + Random Forest")
print("Accuracy:", adasyn_rf_accuracy)
print("Macro F1 Score:", adasyn_rf_f1)

print("Classification Report:")
print(classification_report(y_test, adasyn_rf_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, adasyn_rf_pred))

smote_xgb = Pipeline([
    ("smote", SMOTE(random_state=42)),
    ("model", XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    ))
])

smote_xgb.fit(X_train, y_train)

smote_xgb_pred = smote_xgb.predict(X_test)

smote_xgb_accuracy = accuracy_score(
    y_test,
    smote_xgb_pred
)

smote_xgb_f1 = f1_score(
    y_test,
    smote_xgb_pred,
    average="macro"
)

print("SMOTE + XGBoost")
print("Accuracy:", smote_xgb_accuracy)
print("Macro F1 Score:", smote_xgb_f1)

print("Classification Report:")
print(classification_report(y_test, smote_xgb_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, smote_xgb_pred))

ensemble = VotingClassifier(
    estimators=[
        ("smote_rf", smote_rf),
        ("adasyn_rf", adasyn_rf),
        ("smote_xgb", smote_xgb)
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

print("Ensemble Results")
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

smote_cv = cross_val_score(
    smote_rf,
    X_train,
    y_train,
    cv=cv,
    scoring="f1_macro",
    n_jobs=-1
)

adasyn_cv = cross_val_score(
    adasyn_rf,
    X_train,
    y_train,
    cv=cv,
    scoring="f1_macro",
    n_jobs=-1
)

xgb_cv = cross_val_score(
    smote_xgb,
    X_train,
    y_train,
    cv=cv,
    scoring="f1_macro",
    n_jobs=-1
)

ensemble_cv = cross_val_score(
    ensemble,
    X_train,
    y_train,
    cv=cv,
    scoring="f1_macro",
    n_jobs=-1
)

print("SMOTE + Random Forest CV Macro F1:")
print(smote_cv)
print("Mean:", smote_cv.mean())

print("ADASYN + Random Forest CV Macro F1:")
print(adasyn_cv)
print("Mean:", adasyn_cv.mean())

print("SMOTE + XGBoost CV Macro F1:")
print(xgb_cv)
print("Mean:", xgb_cv.mean())

print("Ensemble CV Macro F1:")
print(ensemble_cv)
print("Mean:", ensemble_cv.mean())

comparison = pd.DataFrame({
    "Model": [
        "SMOTE + Random Forest",
        "ADASYN + Random Forest",
        "SMOTE + XGBoost",
        "Voting Ensemble"
    ],
    "Test Accuracy": [
        smote_rf_accuracy,
        adasyn_rf_accuracy,
        smote_xgb_accuracy,
        ensemble_accuracy
    ],
    "Test Macro F1": [
        smote_rf_f1,
        adasyn_rf_f1,
        smote_xgb_f1,
        ensemble_f1
    ],
    "CV Macro F1": [
        smote_cv.mean(),
        adasyn_cv.mean(),
        xgb_cv.mean(),
        ensemble_cv.mean()
    ]
})

print("Final Model Comparison:")
print(comparison)

results = pd.DataFrame({
    "Actual Class": y_test.values,
    "SMOTE Random Forest": smote_rf_pred,
    "ADASYN Random Forest": adasyn_rf_pred,
    "SMOTE XGBoost": smote_xgb_pred,
    "Ensemble": ensemble_pred
})

print("Prediction Results:")
print(results.head(20))
