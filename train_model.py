import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import joblib

DATA_PATH = "data/machine_data.csv"
MODEL_DIR = "models"
EDA_DIR = "outputs/eda"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EDA_DIR, exist_ok=True)

# XGBoost is optional — use it if available, skip gracefully otherwise
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape}")
print(df.head())

# ---------------------------------------------------------------
# 2. Clean data
# ---------------------------------------------------------------
print("\nCleaning data...")
before = len(df)
df = df.drop_duplicates()
print(f"Removed {before - len(df)} duplicate rows")

print("Missing values before imputation:")
print(df.isnull().sum())

numeric_cols = ["Temperature", "Vibration", "RPM", "Voltage", "Current", "Pressure"]
for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

df = df.dropna(subset=["Status"])  # target must always be present

# ---------------------------------------------------------------
# 3. EDA (saved to outputs/eda/*.png)
# ---------------------------------------------------------------
print("\nRunning EDA and saving charts...")
sns.set_style("whitegrid")

# Class distribution (bar chart)
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Status", order=["Healthy", "Warning", "Faulty"])
plt.title("Machine Status Distribution")
plt.tight_layout()
plt.savefig(f"{EDA_DIR}/status_distribution.png")
plt.close()

# Correlation heatmap
plt.figure(figsize=(7, 5))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{EDA_DIR}/correlation_heatmap.png")
plt.close()

# Histograms of key sensors
df[numeric_cols].hist(figsize=(10, 6), bins=30)
plt.tight_layout()
plt.savefig(f"{EDA_DIR}/feature_histograms.png")
plt.close()

# Pie chart of machine type distribution
plt.figure(figsize=(5, 5))
df["Machine_Type"].value_counts().plot.pie(autopct="%1.1f%%")
plt.title("Machine Type Distribution")
plt.ylabel("")
plt.tight_layout()
plt.savefig(f"{EDA_DIR}/machine_type_pie.png")
plt.close()

print(f"EDA charts saved to {EDA_DIR}/")

# ---------------------------------------------------------------
# 4. Feature engineering + encoding + scaling
# ---------------------------------------------------------------
print("\nEncoding categorical features...")
le_machine = LabelEncoder()
df["Machine_Type_Enc"] = le_machine.fit_transform(df["Machine_Type"])

le_status = LabelEncoder()
df["Status_Enc"] = le_status.fit_transform(df["Status"])
# Keep track of label order for the app: le_status.classes_

feature_cols = [
    "Machine_Type_Enc",
    "Temperature",
    "Vibration",
    "RPM",
    "Voltage",
    "Current",
    "Pressure",
]

X = df[feature_cols]
y = df["Status_Enc"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------------------
# 5. Train/test split
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# ---------------------------------------------------------------
# 6. Train multiple models
# ---------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": SVC(probability=True, random_state=42),
}
if HAS_XGB:
    models["XGBoost"] = XGBClassifier(
        use_label_encoder=False, eval_metric="mlogloss", random_state=42
    )

results = []
trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="weighted", zero_division=0)
    rec = recall_score(y_test, preds, average="weighted", zero_division=0)
    f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

    results.append(
        {"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}
    )
    trained_models[name] = model
    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

results_df = pd.DataFrame(results).sort_values("F1", ascending=False)
print("\n=== Model Comparison ===")
print(results_df.to_string(index=False))
results_df.to_csv("outputs/model_comparison.csv", index=False)

# ---------------------------------------------------------------
# 7. Pick best model (Random Forest recommended; fallback to best F1)
# ---------------------------------------------------------------
best_name = "Random Forest" if "Random Forest" in trained_models else results_df.iloc[0]["Model"]
best_model = trained_models[best_name]
print(f"\nSelected best model: {best_name}")

preds = best_model.predict(X_test)
print("\nClassification Report (best model):")
print(classification_report(y_test, preds, target_names=le_status.classes_))

cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(5, 4))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=le_status.classes_, yticklabels=le_status.classes_,
)
plt.title(f"Confusion Matrix - {best_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{EDA_DIR}/confusion_matrix.png")
plt.close()

# ---------------------------------------------------------------
# 8. Save model + scaler + encoders with joblib
# ---------------------------------------------------------------
joblib.dump(best_model, f"{MODEL_DIR}/best_model.pkl")
joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
joblib.dump(le_machine, f"{MODEL_DIR}/le_machine.pkl")
joblib.dump(le_status, f"{MODEL_DIR}/le_status.pkl")
joblib.dump(feature_cols, f"{MODEL_DIR}/feature_cols.pkl")
joblib.dump(best_name, f"{MODEL_DIR}/model_name.pkl")

print(f"\nSaved model artifacts to {MODEL_DIR}/")
print("Training complete. You can now run: streamlit run app.py")
