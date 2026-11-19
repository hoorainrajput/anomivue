"""
ANOMIVUE Intelligence - compare_models.py  (P1)

Sab models ko ek hi test par chalata hai aur compare karta hai.

Fark evaluate.py se:
  - Split TIME ke hisaab se hai (pehle 70% din = train, baaki 30% = test),
    asli zindagi ki tarah: purane data se seekho, naye par parkho.
  - Zyada models aur features compare hote hain.

Chalane ka tareeqa:   python compare_models.py
Output:               results/model_comparison.csv, results/model_recall_by_type.csv

NOTE: Injected anomalies "guna zyada / guna kam" (multiplicative) type ke hain. Log features
isi tarah ki tabdeeli ko asaani se pakadte hain. Isliye yeh numbers is test ke liye hain,
asli duniya ki anomalies ki guarantee nahi.
"""
import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from config import CONTAMINATION
from evaluate import inject_anomalies, ANOMALY_TYPES
from models import (RAW_FEATURES, LOG_FEATURES, TIME_FEATURES, load_data,
                    add_log_features, add_time_features,
                    ZScoreBaseline, IForest, LOF, OCSVM, RankEnsemble)

SEEDS = [0, 1, 2, 3, 4]
TRAIN_FRACTION = 0.7


def build_models(seed):
    """name -> (feature list, model)"""
    return {
        "ZScore (raw)":            (RAW_FEATURES, ZScoreBaseline()),
        "IsolationForest (raw)":   (RAW_FEATURES, IForest(seed)),
        "LOF (raw)":               (RAW_FEATURES, LOF()),
        "IsolationForest (log)":   (LOG_FEATURES, IForest(seed)),
        "LOF (log)":               (LOG_FEATURES, LOF()),
        "OneClassSVM (log)":       (LOG_FEATURES, OCSVM(seed=seed)),
        "Ensemble IF+LOF+SVM (log)": (LOG_FEATURES, RankEnsemble([IForest(seed), LOF(), OCSVM(seed=seed)])),
        "LOF (log + time)":        (LOG_FEATURES + TIME_FEATURES, LOF()),
    }


def main():
    data = load_data()
    cut_date = data["date"].quantile(TRAIN_FRACTION)
    train = data[data["date"] <= cut_date].reset_index(drop=True)
    test = data[data["date"] > cut_date].reset_index(drop=True)

    rows, type_rows = [], []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        test_inj, y, kinds = inject_anomalies(test, rng)

        # Features: train + test saath mein, taake time features pichle din ka data dekh saken
        combined = pd.concat([train, test_inj], ignore_index=True)
        combined = add_time_features(add_log_features(combined))
        tr_f, te_f = combined.iloc[:len(train)], combined.iloc[len(train):]

        for name, (cols, model) in build_models(seed).items():
            model.fit(tr_f[cols])
            scores = model.score(te_f[cols])
            pred = model.predict(te_f[cols], CONTAMINATION)

            prec, rec, _ = precision_recall_curve(y, scores)
            best_f1 = float((2 * prec * rec / (prec + rec + 1e-12)).max())
            tp = int((pred & (y == 1)).sum())
            rows.append({
                "model": name, "seed": seed,
                "pr_auc": average_precision_score(y, scores),
                "best_f1": best_f1,
                "precision": tp / max(int(pred.sum()), 1),
                "recall": tp / int(y.sum()),
                "false_alarm_rate": int((pred & (y == 0)).sum()) / int((y == 0).sum()),
            })
            for kind in ANOMALY_TYPES:
                type_rows.append({"model": name, "seed": seed, "type": kind,
                                  "recall": float(pred[kinds == kind].mean())})

    order = list(build_models(0).keys())
    res = pd.DataFrame(rows)
    summary = res.groupby("model")[["pr_auc", "best_f1", "precision", "recall", "false_alarm_rate"]].mean()
    summary["pr_auc_std"] = res.groupby("model")["pr_auc"].std()
    summary = summary.reindex(order).round(3).reset_index()

    by_type = (pd.DataFrame(type_rows).groupby(["model", "type"])["recall"].mean()
               .unstack().reindex(order).round(2).reset_index())

    os.makedirs("results", exist_ok=True)
    summary.to_csv("results/model_comparison.csv", index=False)
    by_type.to_csv("results/model_recall_by_type.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"Train: {len(train)} rows (<= {cut_date.date()}), Test: {len(test)} rows, "
          f"injected: {int(y.sum())}, seeds: {len(SEEDS)}, threshold contamination: {CONTAMINATION}\n")
    print("=== Model comparison (mean over seeds, sorted as listed) ===")
    print(summary.to_string(index=False))
    print("\n=== Recall by anomaly type ===")
    print(by_type.to_string(index=False))
    print("\nSaved: results/model_comparison.csv, results/model_recall_by_type.csv")


if __name__ == "__main__":
    main()
