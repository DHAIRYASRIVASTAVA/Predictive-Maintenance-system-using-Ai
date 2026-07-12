"""
generate_dataset.py
--------------------
Generates a synthetic "Machine Fault Detection" dataset that mirrors the
structure of common Kaggle predictive-maintenance datasets (Temperature,
Vibration, RPM, Voltage, Current, Pressure, Machine_Type -> Status).

Why synthetic? This lets the whole project run end-to-end immediately.
If you have the real Kaggle "Machine Fault Detection Dataset", just drop
it in as data/machine_data.csv with the same column names and skip this
script — everything downstream (train_model.py, app.py) will work as-is.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

N = 6000
machine_types = ["Motor", "Pump", "Compressor", "Turbine"]

def generate_row(status):
    machine_type = np.random.choice(machine_types)

    if status == "Healthy":
        temperature = np.random.normal(60, 5)
        vibration   = np.random.normal(2.0, 0.4)
        rpm         = np.random.normal(1500, 60)
        voltage     = np.random.normal(220, 4)
        current     = np.random.normal(10, 1)
        pressure    = np.random.normal(5.0, 0.3)
    elif status == "Warning":
        temperature = np.random.normal(75, 6)
        vibration   = np.random.normal(4.0, 0.6)
        rpm         = np.random.normal(1450, 90)
        voltage     = np.random.normal(215, 6)
        current     = np.random.normal(13, 1.5)
        pressure    = np.random.normal(4.3, 0.5)
    else:  # Faulty
        temperature = np.random.normal(92, 7)
        vibration   = np.random.normal(7.0, 1.0)
        rpm         = np.random.normal(1350, 130)
        voltage     = np.random.normal(205, 9)
        current     = np.random.normal(17, 2.2)
        pressure    = np.random.normal(3.4, 0.6)

    return [
        machine_type,
        round(max(temperature, 0), 2),
        round(max(vibration, 0), 3),
        round(max(rpm, 0), 1),
        round(max(voltage, 0), 2),
        round(max(current, 0), 2),
        round(max(pressure, 0), 2),
        status,
    ]

statuses = np.random.choice(
    ["Healthy", "Warning", "Faulty"], size=N, p=[0.55, 0.30, 0.15]
)

rows = [generate_row(s) for s in statuses]

df = pd.DataFrame(
    rows,
    columns=[
        "Machine_Type",
        "Temperature",
        "Vibration",
        "RPM",
        "Voltage",
        "Current",
        "Pressure",
        "Status",
    ],
)

# introduce a few missing values / duplicates on purpose, so the
# preprocessing step in train_model.py has real cleaning work to do
mask = df.sample(frac=0.01, random_state=1).index
df.loc[mask, "Pressure"] = np.nan
df = pd.concat([df, df.sample(20, random_state=2)], ignore_index=True)

os.makedirs("data", exist_ok=True)
df.to_csv("data/machine_data.csv", index=False)

print(f"Generated data/machine_data.csv with {len(df)} rows")
print(df["Status"].value_counts())
