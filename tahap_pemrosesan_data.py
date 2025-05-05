import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import time

# Load dataset
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# Preprocessing
df.drop("id", axis=1, inplace=True)
df['bmi'] = df['bmi'].fillna(df['bmi'].median())

# Encode categorical data
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Split features and target
X = df.drop("stroke", axis=1)
y = df["stroke"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Record start time
start_time = time.time()

# Train the model
model.fit(X_train, y_train)

# Record end time
end_time = time.time()

# Calculate training duration
training_time = end_time - start_time
print(f"Waktu pelatihan model: {training_time:.2f} detik")

# Evaluate model
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Print classification report
print(classification_report(y_test, y_pred))

# Create folder for visualizations
os.makedirs("visualisasi", exist_ok=True)

# Confusion Matrix visualization
plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", xticklabels=["No Stroke", "Stroke"], yticklabels=["No Stroke", "Stroke"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("visualisasi/confusion_matrix.png")
plt.close()

# ROC Curve visualization
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc_score(y_test, y_prob):.2f}")
plt.plot([0,1], [0,1], color="navy", lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.savefig("visualisasi/roc_curve.png")
plt.close()

# Feature Importance visualization
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns
plt.figure(figsize=(10,6))
sns.barplot(x=importances[indices], y=features[indices])
plt.title("Feature Importance - Random Forest")
plt.tight_layout()
plt.savefig("visualisasi/feature_importance.png")
plt.close()

# Print model evaluation results
print("=== Evaluasi Model Random Forest ===")
print(f"Akurasi     : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision   : {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall      : {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1-Score    : {f1_score(y_test, y_pred, zero_division=0):.4f}")
print(f"AUC Score   : {roc_auc_score(y_test, y_prob):.4f}")
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, zero_division=0))
