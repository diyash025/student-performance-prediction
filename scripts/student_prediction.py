# Student Performance Prediction Using Machine Learning

import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Create outputs folder if it does not exist
os.makedirs("outputs", exist_ok=True)


# 1. Load dataset
df = pd.read_csv("data/student-mat.csv", sep=";")

print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum())


# 2. Create target column
# G3 is the final grade. If G3 >= 10, student passed. Otherwise, failed.
df["result"] = df["G3"].apply(lambda x: 1 if x >= 10 else 0)

print("\nPass/Fail count:")
print(df["result"].value_counts())


# 3. Select input features and target
X = df.drop(["G3", "result"], axis=1)
y = df["result"]


# 4. Convert categorical/text columns into numbers
X = pd.get_dummies(X, drop_first=True)


# 5. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 6. Train Logistic Regression model
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

log_predictions = log_model.predict(X_test)

print("\n--- Logistic Regression Results ---")
print("Accuracy:", accuracy_score(y_test, log_predictions))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, log_predictions))
print("\nClassification Report:")
print(classification_report(y_test, log_predictions))


# 7. Train Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

print("\n--- Random Forest Results ---")
print("Accuracy:", accuracy_score(y_test, rf_predictions))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_predictions))
print("\nClassification Report:")
print(classification_report(y_test, rf_predictions))


# 8. Feature importance using Random Forest
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

importance = importance.sort_values(by="Importance", ascending=False)

print("\nTop 10 important features:")
print(importance.head(10))


# 9. Create CSV file for Power BI dashboard

# Create a copy of the original dataset
powerbi_df = df.copy()

# Add student ID so Power BI scatter chart shows one dot per student
powerbi_df["student_id"] = range(1, len(powerbi_df) + 1)

# Predict result for all students using Random Forest model
all_predictions = rf_model.predict(X)

# Convert actual result from 0/1 to Fail/Pass
powerbi_df["actual_result"] = powerbi_df["result"].map({
    1: "Pass",
    0: "Fail"
})

# Convert predicted result from 0/1 to Fail/Pass
powerbi_df["predicted_result"] = pd.Series(all_predictions).map({
    1: "Pass",
    0: "Fail"
})

# Check whether prediction was correct or wrong
powerbi_df["prediction_status"] = powerbi_df.apply(
    lambda row: "Correct" if row["actual_result"] == row["predicted_result"] else "Wrong",
    axis=1
)

# Keep useful columns for Power BI
powerbi_export = powerbi_df[
    [
        "student_id",
        "school",
        "sex",
        "age",
        "studytime",
        "failures",
        "absences",
        "internet",
        "higher",
        "G1",
        "G2",
        "G3",
        "actual_result",
        "predicted_result",
        "prediction_status"
    ]
]

# Save the CSV file
powerbi_export.to_csv("outputs/student_predictions.csv", index=False)

print("\nPower BI CSV file created successfully:")
print("outputs/student_predictions.csv")