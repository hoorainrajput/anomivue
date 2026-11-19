import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

from config import (FEATURES, DATA_PATH, MODEL_PATH,
                    CONTAMINATION, N_ESTIMATORS, RANDOM_STATE)

# Load processed data
df = pd.read_csv(DATA_PATH)

# Features for anomaly detection
features = df[FEATURES]

# Train model
model = IsolationForest(
    n_estimators=N_ESTIMATORS,
    contamination=CONTAMINATION,
    random_state=RANDOM_STATE
)

model.fit(features)

# Save model
joblib.dump(model, MODEL_PATH)

print(f"Model trained and saved to {MODEL_PATH} (contamination={CONTAMINATION})")