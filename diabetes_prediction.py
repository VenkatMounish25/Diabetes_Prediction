import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

print("DIABETES PREDICTION SYSTEM")

df = pd.read_csv("diabetes_binary_health.csv")

print("\nDataset Shape:")
print(df.shape)

print("\nFirst Five Records:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDataset Summary:")
print(df.describe())

target = "Diabetes_binary"

X = df.drop(columns=[target])
y = df[target]

X = X.fillna(X.mean())

print("\nNumber of Features:", len(X.columns))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples :", len(X_test))

model = RandomForestClassifier(
    n_estimators=20,
    random_state=42
)

print("\nTraining Model...")
model.fit(X_train, y_train)
print("Training Completed")
joblib.dump(model, "diabetes_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Model Saved Successfully!")
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print("\nTraining Accuracy:", train_accuracy)
print("Testing Accuracy :", test_accuracy)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)

print("\nModel Evaluation")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)
print("ROC AUC  :", roc)

print("\nClassification Report")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix")
print(cm)

tn, fp, fn, tp = cm.ravel()

print("\nTrue Negatives :", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives :", tp)

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features")
print(feature_importance.head(10))

print("\nEnter Patient Details")

high_bp = float(input("High Blood Pressure (0/1): "))
high_chol = float(input("High Cholesterol (0/1): "))
bmi = float(input("BMI: "))
smoker = float(input("Smoker (0/1): "))
stroke = float(input("Stroke History (0/1): "))
heart_disease = float(input("Heart Disease (0/1): "))
phys_activity = float(input("Physical Activity (0/1): "))
age = float(input("Age Category (1-13): "))

sample_df = pd.DataFrame(
    [[high_bp, high_chol, bmi, smoker,
      stroke, heart_disease,
      phys_activity, age]],
    columns=X.columns[:8]
)

for col in X.columns[8:]:
    sample_df[col] = 0

sample_df = sample_df[X.columns]

sample_scaled = scaler.transform(sample_df)

prediction = model.predict(sample_scaled)[0]
probability = model.predict_proba(sample_scaled)[0][1]

print("\nPrediction Result")

if prediction == 1:
    print("Result: Diabetic")
else:
    print("Result: Non-Diabetic")

print("Risk Percentage:", round(probability * 100, 2), "%")

print("\nProject Summary")
print("Algorithm Used : Random Forest")
print("Dataset Name   : Diabetes Health Indicators")
print("Total Records  :", len(df))
print("Total Features :", len(X.columns))
print("Model Accuracy :", round(accuracy * 100, 2), "%")

print("\nProgram Executed Successfully")