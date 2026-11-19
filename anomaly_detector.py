import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

from config import (FEATURES, MODEL_PATH,
                    CONTAMINATION, N_ESTIMATORS, RANDOM_STATE)


def train_model(df):
    features = df[FEATURES]

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE
    )
    model.fit(features)

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved! (contamination={CONTAMINATION})")


def detect_anomalies(df):
    model = joblib.load(MODEL_PATH)

    features = df[FEATURES]
    df['anomaly'] = model.predict(features)

    # Convert: -1 -> anomaly, 1 -> normal
    df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})

    return df