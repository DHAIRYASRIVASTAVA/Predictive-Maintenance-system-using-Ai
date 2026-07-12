# 🛠️ Predictive Maintenance using AI

An end-to-end machine learning project that predicts machine health
(**Healthy / Warning / Faulty**) from live sensor readings, estimates
**failure probability**, assigns a **risk level**, and gives a
**maintenance recommendation** — all wrapped in an interactive Streamlit app
with SQLite-backed prediction history.

---

## 📁 Project Structure

```
predictive_maintenance/
├── generate_dataset.py     # Creates a synthetic machine fault dataset (data/machine_data.csv)
├── train_model.py          # Cleans data, runs EDA, trains & compares models, saves best model
├── app.py                  # Streamlit frontend: Predict / History / Dashboard / About
├── requirements.txt
├── README.md
├── data/
│   └── machine_data.csv    # generated dataset (or replace with the real Kaggle CSV)
├── models/                 # saved model, scaler, encoders (joblib)
└── outputs/
    └── eda/                # saved EDA charts (PNG)
```

---

## 🚀 Quick Start (Local / Colab)

### 1. Clone and set up
```bash
git clone https://github.com/<your-username>/predictive-maintenance-ai.git
cd predictive-maintenance-ai
pip install -r requirements.txt
```

### 2. Get the dataset
**Option A — Use the real Kaggle dataset (recommended for a real project):**
- Download the "Machine Fault Detection Dataset" from Kaggle.
- Rename/place it at `data/machine_data.csv`.
- Make sure the columns match: `Machine_Type, Temperature, Vibration, RPM, Voltage, Current, Pressure, Status`
  (rename your columns to match, or edit `train_model.py`/`app.py` accordingly).

**Option B — Use the built-in synthetic generator (works immediately, no download needed):**
```bash
python generate_dataset.py
```

### 3. Train the model
```bash
python train_model.py
```
This will:
- Clean the data (remove duplicates, handle missing values)
- Save EDA charts to `outputs/eda/`
- Train Logistic Regression, Decision Tree, Random Forest, SVM (and XGBoost if installed)
- Print accuracy / precision / recall / F1 / confusion matrix for each
- Save the best model + scaler + encoders to `models/` using `joblib`

### 4. Run the Streamlit app
```bash
streamlit run app.py
```
Open the local URL Streamlit prints (usually `http://localhost:8501`).

---

## 🖥️ Using the App

- **Predict** — enter Machine Type, Temperature, Vibration, RPM, Voltage, Current,
  Pressure → get:
  - Machine Status (Healthy / Warning / Faulty)
  - Failure Probability (%)
  - Risk Level (Low / Medium / High)
  - Maintenance Recommendation
  - Every prediction is saved automatically to a local SQLite database (`predictions.db`)
- **Prediction History** — view/download all past predictions, or clear history
- **Dashboard** — bar chart, pie chart, histogram, line chart, and a correlation
  heatmap built from your prediction history
- **About** — project summary

---

## ☁️ Deploying to Streamlit Community Cloud

1. Push this project to a public GitHub repository (include `data/machine_data.csv`
   and the `models/` folder, or add a build step that runs
   `generate_dataset.py` + `train_model.py` before the app starts).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. Click **New app**, pick your repo/branch, and set the main file to `app.py`.
4. Deploy. Streamlit Cloud installs everything from `requirements.txt` automatically.

> Tip: SQLite on Streamlit Cloud is ephemeral (resets on redeploy/restart).
> That's fine for a demo; for persistence in production, swap SQLite for a
> hosted database.

---

## 🧠 Model Details

| Step | Technique |
|---|---|
| Missing values | Median imputation (numeric columns) |
| Duplicates | Dropped |
| Categorical encoding | `LabelEncoder` (Machine_Type, Status) |
| Scaling | `StandardScaler` |
| Split | 80/20 train-test, stratified |
| Models compared | Logistic Regression, Decision Tree, Random Forest, SVM, (XGBoost optional) |
| Selected model | Random Forest (recommended; script auto-picks best F1 if you swap it out) |
| Evaluation | Accuracy, Precision, Recall, F1, Confusion Matrix, Classification Report |
| Persistence | `joblib` (`models/best_model.pkl`, `scaler.pkl`, encoders) |

---

## 📊 Input Features

| Feature | Description |
|---|---|
| Machine_Type | Motor / Pump / Compressor / Turbine |
| Temperature | °C |
| Vibration | mm/s |
| RPM | Rotational speed |
| Voltage | V |
| Current | A |
| Pressure | bar |

## 🎯 Output

- **Machine Status**: Healthy / Warning / Faulty
- **Failure Probability**: % (derived from model's predicted class probabilities)
- **Risk Level**: Low / Medium / High
- **Maintenance Recommendation**: actionable text guidance

---

## 🔧 Tech Stack

Python · Pandas · NumPy · Matplotlib · Seaborn · Plotly · Scikit-learn ·
XGBoost (optional) · Joblib · Streamlit · SQLite · Git/GitHub

---

## 📌 Notes

- Replace the synthetic dataset with the real Kaggle "Machine Fault Detection
  Dataset" any time — just keep the column names consistent (or adjust the
  scripts).
- To improve accuracy further: try `GridSearchCV`/`RandomizedSearchCV` for
  hyperparameter tuning, add more engineered features (e.g. rolling averages
  if you have time-series sensor logs), or add class-imbalance handling
  (e.g. `class_weight="balanced"`, SMOTE) if your real dataset is skewed.
