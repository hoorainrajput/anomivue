"""
ANOMIVUE Intelligence - evaluate.py  (P0.3)

Masla: dataset mein anomalies ke asli labels nahi hain, isliye Precision/Recall
nikalna mumkin nahi tha. Hal: hum KHUD known anomalies inject karte hain aur
dekhte hain ke model unhe pakadta hai ya nahi.

Tareeqa:
  1. Data ko train (70%) / test (30%) mein baanto.
  2. Model sirf train data par seekhta hai.
  3. Test data mein ~5% rows ko 3 tarah se bigaad do (label = 1), baaki label = 0:
       viral_spike       -> likes/shares/engagement achanak bohat zyada
       bot_burst         -> post_count achanak bohat zyada, engagement kam
       engagement_crash  -> likes/shares/engagement achanak gir jaye
  4. Precision, Recall, F1, PR-AUC nikalo.
  5. 5 alag random seeds par dohrao taake ek lucky run se dhoka na ho.

Chalane ka tareeqa:   python evaluate.py
Output:               results/metrics.csv, results/recall_by_type.csv
"""
import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score)

from config import (DATA_PATH, FEATURES, N_ESTIMATORS)

SEEDS = [0, 1, 2, 3, 4]
CONTAMINATIONS = [0.01, 0.02, 0.05, 0.10]
INJECT_FRACTION = 0.05          # test rows ka kitna hissa bigaadna hai
ANOMALY_TYPES = ["viral_spike", "bot_burst", "engagement_crash"]


# --------------------------------------------------------------------------
# 1. Anomalies inject karna
# --------------------------------------------------------------------------
def inject_anomalies(test_df, rng):
    """Test data ki copy mein anomalies dalta hai. Wapas (df, label, type) deta hai."""
    df = test_df.copy().reset_index(drop=True)
    n = len(df)
    labels = np.zeros(n, dtype=int)
    types = np.array(["normal"] * n, dtype=object)

    n_inject = int(round(n * INJECT_FRACTION))
    chosen = rng.choice(n, size=n_inject, replace=False)
    groups = np.array_split(chosen, len(ANOMALY_TYPES))

    for kind, idx in zip(ANOMALY_TYPES, groups):
        k = len(idx)
        if kind == "viral_spike":
            df.loc[idx, "likes"] = df.loc[idx, "likes"] * rng.uniform(5, 10, k)
            df.loc[idx, "shares"] = df.loc[idx, "shares"] * rng.uniform(5, 10, k)
            df.loc[idx, "engagement_rate"] = df.loc[idx, "engagement_rate"] * rng.uniform(3, 6, k)
        elif kind == "bot_burst":
            df.loc[idx, "post_count"] = rng.integers(6, 13, k)
            df.loc[idx, "engagement_rate"] = df.loc[idx, "engagement_rate"] * rng.uniform(0.2, 0.5, k)
        elif kind == "engagement_crash":
            df.loc[idx, "likes"] = df.loc[idx, "likes"] * rng.uniform(0.0, 0.05, k)
            df.loc[idx, "shares"] = df.loc[idx, "shares"] * rng.uniform(0.0, 0.05, k)
            df.loc[idx, "engagement_rate"] = df.loc[idx, "engagement_rate"] * rng.uniform(0.0, 0.1, k)
        labels[idx] = 1
        types[idx] = kind

    return df, labels, types


# --------------------------------------------------------------------------
# 2. Models
# --------------------------------------------------------------------------
class ZScoreBaseline:
    """Seedha statistical baseline: kisi bhi feature ka |z-score| bohat bada ho to anomaly.
    Agar ML model isko beat nahi karta to ML ki zaroorat hi nahi."""

    def __init__(self, contamination):
        self.contamination = contamination

    def _score(self, X):
        z = (X - self.mean_) / self.std_
        return np.abs(z).max(axis=1).to_numpy()

    def fit(self, X):
        self.mean_ = X.mean()
        self.std_ = X.std().replace(0, 1)
        train_scores = self._score(X)
        self.threshold_ = np.quantile(train_scores, 1 - self.contamination)
        return self

    def score(self, X):
        return self._score(X)          # bada = zyada anomalous

    def predict(self, X):
        return (self.score(X) > self.threshold_).astype(int)


class IForest:
    def __init__(self, contamination, seed):
        self.model = IsolationForest(n_estimators=N_ESTIMATORS,
                                     contamination=contamination,
                                     random_state=seed)

    def fit(self, X):
        self.model.fit(X)
        return self

    def score(self, X):
        return -self.model.decision_function(X)   # bada = zyada anomalous

    def predict(self, X):
        return (self.model.predict(X) == -1).astype(int)


# --------------------------------------------------------------------------
# 3. Evaluation
# --------------------------------------------------------------------------
def main():
    data = pd.read_csv(DATA_PATH)[FEATURES].astype(float)
    metric_rows, type_rows = [], []

    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(data))
        cut = int(len(data) * 0.7)
        train = data.iloc[order[:cut]].reset_index(drop=True)
        test = data.iloc[order[cut:]].reset_index(drop=True)

        test_inj, y_true, kinds = inject_anomalies(test, rng)

        for c in CONTAMINATIONS:
            models = {
                "IsolationForest": IForest(c, seed).fit(train),
                "ZScore baseline": ZScoreBaseline(c).fit(train),
            }
            for name, m in models.items():
                y_pred = m.predict(test_inj)
                scores = m.score(test_inj)
                metric_rows.append({
                    "model": name, "contamination": c, "seed": seed,
                    "precision": precision_score(y_true, y_pred, zero_division=0),
                    "recall": recall_score(y_true, y_pred, zero_division=0),
                    "f1": f1_score(y_true, y_pred, zero_division=0),
                    "pr_auc": average_precision_score(y_true, scores),
                })
                for kind in ANOMALY_TYPES:
                    mask = kinds == kind
                    type_rows.append({
                        "model": name, "contamination": c, "seed": seed,
                        "type": kind,
                        "recall": float(y_pred[mask].mean()),
                    })

    metrics = pd.DataFrame(metric_rows)
    summary = (metrics.groupby(["model", "contamination"])
               [["precision", "recall", "f1", "pr_auc"]]
               .agg(["mean", "std"]).round(3))
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()

    by_type = (pd.DataFrame(type_rows)
               .groupby(["model", "contamination", "type"])["recall"]
               .mean().round(3).reset_index())

    os.makedirs("results", exist_ok=True)
    summary.to_csv("results/metrics.csv", index=False)
    by_type.to_csv("results/recall_by_type.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"\nTest rows har run mein: {len(test_inj)}, injected anomalies: {int(y_true.sum())} "
          f"({INJECT_FRACTION:.0%}), seeds: {len(SEEDS)}\n")
    print("=== Overall (mean +- std over seeds) ===")
    print(summary.to_string(index=False))
    print("\n=== Recall by anomaly type (contamination = 0.05) ===")
    print(by_type[by_type.contamination == 0.05]
          .pivot(index="model", columns="type", values="recall").to_string())
    print("\nSaved: results/metrics.csv, results/recall_by_type.csv")


if __name__ == "__main__":
    main()