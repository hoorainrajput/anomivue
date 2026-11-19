"""
ANOMIVUE Intelligence Configuration File
"""

# Features jin par model chalta hai
FEATURES = ["post_count", "likes", "shares", "engagement_rate"]

# Data aur model ki files
DATA_PATH = "processed_trend_data.csv"
MODEL_PATH = "isolation_forest.pkl"
OUTPUT_PATH = "anomaly_output.csv"

# Isolation Forest ki settings
CONTAMINATION = 0.05   # data ka kitna hissa anomaly maana jaye (yeh andaza hai, asli rate nahi)
N_ESTIMATORS = 100
RANDOM_STATE = 42