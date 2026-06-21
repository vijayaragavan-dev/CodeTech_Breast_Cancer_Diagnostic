"""
Breast Cancer Diagnostic System using Logistic Regression
=========================================================

This program uses Machine Learning to predict whether a breast tumor is
Malignant (Cancerous) or Benign (Non-Cancerous).

It demonstrates the complete ML workflow:
1. Data Collection -> 2. Exploration -> 3. Cleaning -> 4. Visualization
5. Feature Selection -> 6. Train/Test Split -> 7. Model Training
8. Prediction -> 9. Evaluation -> 10. Model Saving

Author: Vijayaragavan U
Dataset: Breast Cancer Wisconsin Dataset (Scikit-Learn)
"""

import os
import sys
import warnings
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120

# =====================================================================
# CREATE OUTPUT DIRECTORIES
# =====================================================================
os.makedirs("screenshots", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

print("=" * 70)
print("BREAST CANCER DIAGNOSTIC SYSTEM")
print("Machine Learning Project - Logistic Regression")
print("=" * 70)

# =====================================================================
# 1. DATASET LOADING
# =====================================================================
print("\n" + "=" * 70)
print("PHASE 1: DATASET LOADING")
print("=" * 70)

try:
    cancer_data = load_breast_cancer()
    print("[OK] Dataset loaded successfully from Scikit-Learn.")
except Exception as e:
    print(f"[ERROR] Failed to load dataset: {e}")
    sys.exit(1)

# Create a Pandas DataFrame with feature columns
feature_names = cancer_data.feature_names
target_names = cancer_data.target_names  # ['malignant', 'benign']

df = pd.DataFrame(cancer_data.data, columns=feature_names)
df["target"] = cancer_data.target

print(f"[INFO] Dataset contains {len(df)} samples with {len(feature_names)} features.")
print(f"[INFO] Target classes: 0 = {target_names[0]}, 1 = {target_names[1]}")

# Save dataset info to a text file
with open("dataset_info.txt", "w", encoding="utf-8") as f:
    f.write("BREAST CANCER WISCONSIN DATASET - INFORMATION\n")
    f.write("=" * 50 + "\n")
    f.write(f"Source: Scikit-Learn (load_breast_cancer)\n")
    f.write(f"Original Source: UCI Machine Learning Repository\n")
    f.write(f"Total Samples: {len(df)}\n")
    f.write(f"Total Features: {len(feature_names)}\n")
    f.write(f"Target Classes: 0 = {target_names[0]}, 1 = {target_names[1]}\n")
    f.write(f"\nFeature Names:\n")
    for i, name in enumerate(feature_names, 1):
        f.write(f"  {i:2d}. {name}\n")

print("[OK] Dataset information saved to dataset_info.txt")

# =====================================================================
# 2. DATASET EXPLORATION
# =====================================================================
print("\n" + "=" * 70)
print("PHASE 2: DATASET EXPLORATION")
print("=" * 70)

print("\n[INFO] First 5 rows of the dataset:")
print(df.head())

print(f"\n[INFO] Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")

print("\n[INFO] Column names and data types:")
print(df.dtypes)

print("\n[INFO] Basic statistical summary:")
print(df.describe())

# =====================================================================
# 3. MISSING VALUE ANALYSIS
# =====================================================================
print("\n" + "=" * 70)
print("PHASE 3: MISSING VALUE ANALYSIS")
print("=" * 70)

missing_values = df.isnull().sum()
total_missing = missing_values.sum()

if total_missing == 0:
    print("[OK] No missing values found in the dataset.")
else:
    print(f"[WARNING] Found {total_missing} missing value(s):")
    print(missing_values[missing_values > 0])

# =====================================================================
# 4. STATISTICAL ANALYSIS
# =====================================================================
print("\n" + "=" * 70)
print("PHASE 4: STATISTICAL ANALYSIS")
print("=" * 70)

print("\n[INFO] Target variable distribution:")
target_counts = df["target"].value_counts()
print(f"  Benign  (Class 1): {target_counts.get(1, 0)} samples")
print(f"  Malignant (Class 0): {target_counts.get(0, 0)} samples")
print(f"\n[INFO] Percentage split:")
print(f"  Benign:    {target_counts.get(1, 0) / len(df) * 100:.2f}%")
print(f"  Malignant: {target_counts.get(0, 0) / len(df) * 100:.2f}%")

print("\n[INFO] Feature statistics grouped by target:")
print(df.groupby("target").describe().T)

# =====================================================================
# 5. DATA VISUALIZATION
# =====================================================================
print("\n" + "=" * 70)
print("PHASE 5: DATA VISUALIZATION")
print("=" * 70)

# --- 5a. Target Distribution Plot ---
print("\n[INFO] Generating target distribution plot...")
plt.figure(figsize=(8, 5))
sns.countplot(
    x="target",
    data=df,
    palette=["#e74c3c", "#2ecc71"],
    edgecolor="black",
)
plt.title("Target Variable Distribution", fontsize=15, fontweight="bold")
plt.xlabel("Tumor Type (0 = Malignant, 1 = Benign)", fontsize=12)
plt.ylabel("Number of Samples", fontsize=12)
plt.xticks(ticks=[0, 1], labels=["Malignant", "Benign"])
plt.tight_layout()
plt.savefig("screenshots/distribution.png", dpi=200, bbox_inches="tight")
print("[OK] Saved: screenshots/distribution.png")
plt.show(block=False)
plt.pause(2)
plt.close()

# --- 5b. Correlation Heatmap ---
print("\n[INFO] Generating correlation heatmap...")
plt.figure(figsize=(16, 12))
correlation_matrix = df.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(
    correlation_matrix,
    mask=mask,
    cmap="coolwarm",
    center=0,
    annot=False,
    fmt=".2f",
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
)
plt.title("Feature Correlation Heatmap", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig("screenshots/heatmap.png", dpi=200, bbox_inches="tight")
print("[OK] Saved: screenshots/heatmap.png")
plt.show(block=False)
plt.pause(2)
plt.close()

# =====================================================================
# 6. FEATURE SELECTION & DATA PREPARATION
# =====================================================================
print("\n" + "=" * 70)
print("PHASE 6: FEATURE SELECTION & DATA PREPARATION")
print("=" * 70)

# Separate features (X) and target (y)
X = df.drop("target", axis=1)
y = df["target"]

print(f"[INFO] Feature matrix (X) shape: {X.shape}")
print(f"[INFO] Target vector (y) shape: {y.shape}")

# Feature scaling — important for Logistic Regression convergence
print("\n[INFO] Applying StandardScaler to normalize features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
print("[OK] Feature scaling completed. All features now have mean=0, std=1.")

# =====================================================================
# 7. TRAIN-TEST SPLIT
# =====================================================================
print("\n" + "=" * 70)
print("PHASE 7: TRAIN-TEST SPLIT")
print("=" * 70)

TEST_SIZE = 0.2
RANDOM_STATE = 42

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print(f"[INFO] Training set:   {X_train.shape[0]} samples ({100 * (1 - TEST_SIZE):.0f}%)")
print(f"[INFO] Testing set:    {X_test.shape[0]} samples ({100 * TEST_SIZE:.0f}%)")
print(f"[INFO] Stratified split ensures class balance in both sets.")

# =====================================================================
# 8. MODEL TRAINING
# =====================================================================
print("\n" + "=" * 70)
print("PHASE 8: MODEL TRAINING")
print("=" * 70)

model = LogisticRegression(max_iter=10000, random_state=RANDOM_STATE)

print("[INFO] Training Logistic Regression model...")
model.fit(X_train, y_train)
print("[OK] Model training completed successfully!")
print(f"[INFO] Number of iterations required: {model.n_iter_[0]}")

# =====================================================================
# 9. MODEL PREDICTION
# =====================================================================
print("\n" + "=" * 70)
print("PHASE 9: MODEL PREDICTION")
print("=" * 70)

print("[INFO] Making predictions on the test set...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("[OK] Predictions generated for all test samples.")

# Display first 10 predictions vs actual
print("\n[INFO] Sample predictions (first 10 test samples):")
comparison = pd.DataFrame(
    {"Actual": y_test.values[:10], "Predicted": y_pred[:10]}
)
comparison["Correct"] = comparison["Actual"] == comparison["Predicted"]
print(comparison)

# =====================================================================
# 10. MODEL EVALUATION
# =====================================================================
print("\n" + "=" * 70)
print("PHASE 10: MODEL EVALUATION")
print("=" * 70)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=target_names)

print(f"\nAccuracy:  {accuracy:.4f}  ({accuracy * 100:.2f}%)")
print(f"Precision: {precision:.4f}  ({precision * 100:.2f}%)")
print(f"Recall:    {recall:.4f}  ({recall * 100:.2f}%)")
print(f"F1 Score:  {f1:.4f}  ({f1 * 100:.2f}%)")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(report)

# --- 10a. Confusion Matrix Heatmap ---
print("\n[INFO] Generating confusion matrix heatmap...")
plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=target_names,
    yticklabels=target_names,
    cbar=False,
    linewidths=1,
    linecolor="white",
)
plt.title("Confusion Matrix - Logistic Regression", fontsize=14, fontweight="bold")
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.tight_layout()
plt.savefig("screenshots/confusion_matrix.png", dpi=200, bbox_inches="tight")
print("[OK] Saved: screenshots/confusion_matrix.png")
plt.show(block=False)
plt.pause(2)
plt.close()

# --- 10b. Save metrics to file ---
metrics_path = "outputs/model_metrics.txt"
with open(metrics_path, "w", encoding="utf-8") as f:
    f.write("=" * 55 + "\n")
    f.write("  BREAST CANCER DIAGNOSTIC SYSTEM\n")
    f.write("  MODEL EVALUATION METRICS\n")
    f.write("=" * 55 + "\n\n")
    f.write(f"Algorithm: Logistic Regression\n")
    f.write(f"Training samples: {X_train.shape[0]}\n")
    f.write(f"Testing samples:  {X_test.shape[0]}\n")
    f.write(f"Test size:        {TEST_SIZE * 100:.0f}%\n\n")
    f.write("-" * 55 + "\n")
    f.write("  PERFORMANCE METRICS\n")
    f.write("-" * 55 + "\n")
    f.write(f"  Accuracy  : {accuracy:.4f}  ({accuracy * 100:.2f}%)\n")
    f.write(f"  Precision : {precision:.4f}  ({precision * 100:.2f}%)\n")
    f.write(f"  Recall    : {recall:.4f}  ({recall * 100:.2f}%)\n")
    f.write(f"  F1 Score  : {f1:.4f}  ({f1 * 100:.2f}%)\n\n")
    f.write("-" * 55 + "\n")
    f.write("  CONFUSION MATRIX\n")
    f.write("-" * 55 + "\n")
    f.write(f"                  Predicted\n")
    f.write(f"                  Malignant  Benign\n")
    f.write(f"  Actual Malignant   {cm[0,0]:3d}       {cm[0,1]:3d}\n")
    f.write(f"         Benign      {cm[1,0]:3d}       {cm[1,1]:3d}\n\n")
    f.write("-" * 55 + "\n")
    f.write("  CLASSIFICATION REPORT\n")
    f.write("-" * 55 + "\n")
    f.write(report)

print(f"[OK] Model metrics saved to: {metrics_path}")

# =====================================================================
# 11. MODEL PERSISTENCE
# =====================================================================
print("\n" + "=" * 70)
print("PHASE 11: MODEL PERSISTENCE (SAVING)")
print("=" * 70)

model_path = "cancer_model.pkl"
try:
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print("[OK] Machine Learning model saved successfully.")
    print(f"[INFO] Model file: {model_path}")
    print(f"[INFO] Model size: {os.path.getsize(model_path) / 1024:.2f} KB")
except Exception as e:
    print(f"[ERROR] Failed to save model: {e}")

# =====================================================================
# 12. PROJECT COMPLETION SUMMARY
# =====================================================================
print("\n" + "=" * 70)
print("PROJECT COMPLETION SUMMARY")
print("=" * 70)

print("""
  BREAST CANCER DIAGNOSTIC SYSTEM - SUMMARY
  -----------------------------------------
  Dataset     : Breast Cancer Wisconsin
  Algorithm   : Logistic Regression
  Accuracy    : {:.2f}%
  Precision   : {:.2f}%
  Recall      : {:.2f}%
  F1 Score    : {:.2f}%

  Files Generated:
  1. cancer_model.pkl        - Trained ML model
  2. dataset_info.txt        - Dataset information
  3. outputs/model_metrics.txt - Evaluation metrics
  4. screenshots/distribution.png - Target distribution plot
  5. screenshots/heatmap.png      - Correlation heatmap
  6. screenshots/confusion_matrix.png - Confusion matrix

  The model can now be used to predict whether a breast tumor
  is Malignant (cancerous) or Benign (non-cancerous).
""".format(
    accuracy * 100, precision * 100, recall * 100, f1 * 100
))

print("=" * 70)
print("Thank you for using the Breast Cancer Diagnostic System!")
print("=" * 70)
