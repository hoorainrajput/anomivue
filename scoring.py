"""
ANOMIVUE Intelligence - scoring.py

Asli (production) model: LOF (Local Outlier Factor) on log-features.

LOF kya karta hai: har point ke liye dekhta hai ke uske 35 sabse qareebi padosi kitne
"ghane" (dense) hain, aur point khud uske muqable mein kitna akela hai.
  LOF ~ 1  -> normal (padosiyon jaisa)
  LOF >> 1 -> anomaly (padosiyon se door, akela)

Severity (0-100) ka formula:   severity = 100 * (1 - 2 ** (-(LOF - 1) / (threshold - 1)))
  -> 0 = bilkul normal, 50 = anomaly ki hadd (threshold). Threshold jitni doori har baar
     badhne par severity ka bacha hua hissa aadha rah jata hai (75, 87.5, 93.8 ...), isliye
     bohat extreme points bhi 100 par ek saath nahi chipakte.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd

from config import (CONTAMINATION, DATASET_JSON_PATHS, MODEL_JSON_PATH,
                    MODEL_PATH, N_NEIGHBORS, OUTPUT_PATH)
from models import LOF, LOG_FEATURES, RAW_FEATURES, add_log_features

DEFAULT_NORMAL_REASON = "Normal behavior"


# --------------------------------------------------------------------------
# Training / saving
# --------------------------------------------------------------------------
def _log_frame(df):
    return add_log_features(df.astype({c: float for c in RAW_FEATURES}))[LOG_FEATURES]


def train_production(df):
    """Poore data par LOF seekho aur threshold tay karo."""
    X = _log_frame(df)
    lof = LOF(n_neighbors=N_NEIGHBORS).fit(X)
    train_lof = lof.lof_ratio(X)
    threshold = float(np.quantile(train_lof, 1 - CONTAMINATION))
    return {
        "lof": lof,
        "threshold": threshold,
        "contamination": CONTAMINATION,
        "n_neighbors": N_NEIGHBORS,
        "n_train": int(len(X)),
        "log_features": LOG_FEATURES,
        "medians": {c: float(df[c].median()) for c in RAW_FEATURES},
    }


def save_bundle(bundle, path=MODEL_PATH):
    joblib.dump(bundle, path)


def load_bundle(path=MODEL_PATH):
    return joblib.load(path)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def lof_scores(bundle, df):
    return bundle["lof"].lof_ratio(_log_frame(df))


def severity(lof, threshold):
    lof = np.asarray(lof, dtype=float)
    if threshold <= 1:
        return np.zeros_like(lof)
    x = np.maximum(lof - 1.0, 0.0) / (threshold - 1.0)
    return (100.0 * (1.0 - np.power(2.0, -x))).round(1)


def _ratio_text(x):
    return f"{x:.0f}x" if x >= 10 else (f"{x:.1f}x" if x >= 1 else f"{x:.2f}x")


def explain(bundle, post_count, likes, shares, engagement_rate):
    """Kaunsa metric sabse zyada alag tha. 'typical' = poore data ka median."""
    row = pd.DataFrame([{"post_count": post_count, "likes": likes, "shares": shares,
                         "engagement_rate": engagement_rate}])
    z = bundle["lof"].scaler.transform(_log_frame(row))[0]       # standardised log features
    names = ["post_count", "likes", "shares", "engagement_rate"]
    vals = {"post_count": post_count, "likes": likes, "shares": shares,
            "engagement_rate": engagement_rate}
    labels = {"post_count": "Post volume", "likes": "Likes", "shares": "Shares",
              "engagement_rate": "Engagement rate"}
    parts = []
    for i in np.argsort(-np.abs(z)):
        if abs(z[i]) < 2 or len(parts) == 2:
            break
        n = names[i]
        med = bundle["medians"][n]
        direction = "high" if z[i] > 0 else "low"
        if n == "post_count":
            parts.append(f"{labels[n]} {direction} ({int(round(vals[n]))} vs typical {int(round(med))})")
        else:
            ratio = vals[n] / med if med > 0 else float("inf")
            parts.append(f"{labels[n]} unusually {direction} ({_ratio_text(ratio)} typical)")
    return "; ".join(parts) if parts else "Unusual combination of metrics (no single metric extreme)"


def annotate(df, bundle=None):
    """df mein lof_score, severity, anomaly (0/1) aur reason columns jodta hai."""
    bundle = bundle or load_bundle()
    out = df.copy()
    lof = lof_scores(bundle, out)
    out["lof_score"] = lof.round(4)
    out["severity"] = severity(lof, bundle["threshold"])
    out["anomaly"] = (lof > bundle["threshold"]).astype(int)
    out["reason"] = [
        explain(bundle, r.post_count, r.likes, r.shares, r.engagement_rate) if a else DEFAULT_NORMAL_REASON
        for r, a in zip(out.itertuples(), out["anomaly"])
    ]
    return out


# --------------------------------------------------------------------------
# Dashboard ke liye export
# --------------------------------------------------------------------------
def export_model_json(bundle, path=MODEL_JSON_PATH):
    """Model ko JSON mein likhta hai taake Next.js (Vercel) bhi bina Python ke
    EXACT yehi LOF score kar sake (src/lib/lof.ts)."""
    sk = bundle["lof"].model
    sc = bundle["lof"].scaler
    payload = {
        "algorithm": "Local Outlier Factor (novelty=True) on log features",
        "n_neighbors": int(sk.n_neighbors_),
        "contamination": bundle["contamination"],
        "threshold": bundle["threshold"],
        "n_train": bundle["n_train"],
        "log_features": bundle["log_features"],
        "medians": bundle["medians"],
        "scaler_mean": [float(v) for v in sc.mean_],
        "scaler_scale": [float(v) for v in sc.scale_],
        "kdist": [round(float(v), 7) for v in sk._distances_fit_X_[:, sk.n_neighbors_ - 1]],
        "lrd": [round(float(v), 7) for v in sk._lrd],
        "train": [[round(float(v), 6) for v in r] for r in sk._fit_X],
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    return path


def export_dashboard_json(annotated, paths=DATASET_JSON_PATHS):
    rows = []
    for i, r in enumerate(annotated.itertuples(), start=1):
        rows.append({
            "id": i,
            "date": str(r.date)[:10],
            "platform": r.platform,
            "hashtag": r.hashtag,
            "post_count": int(r.post_count),
            "likes": int(r.likes),
            "shares": int(r.shares),
            "comments": int(r.comments),
            "engagement_rate": float(r.engagement_rate),
            "z_score": float(r.z_score),
            "is_anomaly": bool(r.anomaly),
            "anomaly_score": float(r.severity),
            "lof_score": float(r.lof_score),
            "reason": r.reason,
        })
    for p in paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            json.dump(rows, f)
    return rows


def export_all(df, bundle):
    """Ek call mein: anomaly_output.csv + dataset.json (2 jagah) + model.json"""
    annotated = annotate(df, bundle)
    annotated.to_csv(OUTPUT_PATH, index=False)
    export_dashboard_json(annotated)
    export_model_json(bundle)
    return annotated
