"""
ANOMIVUE Intelligence - models.py  (P1)

Is file mein do cheezein hain:
  1. Features banane ke functions (raw, log, aur time-aware)
  2. Anomaly models (sab ka interface ek jaisa: fit(X) aur score(X), bada score = zyada anomalous)

Models:
  ZScoreBaseline   - seedha statistical baseline (ML ko isse behtar hona chahiye)
  IForest          - Isolation Forest
  LOF              - Local Outlier Factor (padosi points se density compare karta hai)
  OCSVM            - One-Class SVM
  RankEnsemble     - upar ke models ke scores ka average
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from config import N_ESTIMATORS, DATA_PATH

NUMERIC = ["post_count", "likes", "shares", "comments", "engagement_rate"]
RAW_FEATURES = ["post_count", "likes", "shares", "engagement_rate"]
LOG_FEATURES = ["post_count", "log_likes", "log_shares", "log_engagement_rate"]
TIME_FEATURES = ["dev_post_count", "dev_likes", "dev_shares"]
GROUP_KEYS = ["platform", "hashtag"]


# --------------------------------------------------------------------------
# Data + features
# --------------------------------------------------------------------------
def load_data(path=DATA_PATH):
    """CSV padho, numbers ko float banao, date ke hisaab se sort karo."""
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[["date", "platform", "hashtag"] + NUMERIC].copy()
    df[NUMERIC] = df[NUMERIC].astype(float)
    return df.sort_values("date").reset_index(drop=True)


def add_log_features(df):
    """likes/shares/engagement bohat skewed hain (kuch rows bohat bade). Log lene se
    'kitna guna zyada' wali tabdeeli saaf nazar aati hai, chahe value chhoti ho ya badi."""
    out = df.copy()
    out["log_likes"] = np.log1p(out["likes"])
    out["log_shares"] = np.log1p(out["shares"])
    out["log_engagement_rate"] = np.log1p(out["engagement_rate"])
    return out


def add_time_features(df, window=10, min_periods=3):
    """Har platform+hashtag ke liye: aaj ki value apne PICHLE `window` din ke average se
    kitni door hai (sirf pichla data, aaj ka nahi, taake leak na ho)."""
    out = df.copy()
    order = out.sort_values(GROUP_KEYS + ["date"])
    for col in ["post_count", "likes", "shares"]:
        prev = order.groupby(GROUP_KEYS)[col].shift(1)
        grp = [order["platform"], order["hashtag"]]
        mean = prev.groupby(grp).transform(lambda s: s.rolling(window, min_periods=min_periods).mean())
        std = prev.groupby(grp).transform(lambda s: s.rolling(window, min_periods=min_periods).std())
        dev = ((order[col] - mean) / (std + 1e-6)).clip(-10, 10).fillna(0.0)
        out["dev_" + col] = dev.reindex(out.index)
    return out


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
class _Base:
    """Common kaam: train scores yaad rakhna aur threshold nikalna."""

    def threshold(self, contamination):
        return float(np.quantile(self.train_scores_, 1 - contamination))

    def predict(self, X, contamination):
        return (self.score(X) > self.threshold(contamination)).astype(int)


class ZScoreBaseline(_Base):
    """Kisi bhi feature ka |z-score| bohat bada ho to anomaly."""

    def fit(self, X):
        X = pd.DataFrame(X)
        self.mean_, self.std_ = X.mean(), X.std().replace(0, 1)
        self.train_scores_ = self.score(X)
        return self

    def score(self, X):
        z = (pd.DataFrame(X) - self.mean_) / self.std_
        return np.abs(z).max(axis=1).to_numpy()


class IForest(_Base):
    def __init__(self, seed=42, n_estimators=N_ESTIMATORS):
        self.model = IsolationForest(n_estimators=n_estimators, random_state=seed)

    def fit(self, X):
        self.model.fit(X)
        self.train_scores_ = self.score(X)
        return self

    def score(self, X):
        return -self.model.decision_function(X)


class LOF(_Base):
    def __init__(self, n_neighbors=35):
        self.scaler = StandardScaler()
        self.model = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)

    def fit(self, X):
        A = self.scaler.fit_transform(X)
        self.model.fit(A)
        self.train_scores_ = -self.model.decision_function(A)
        return self

    def score(self, X):
        return -self.model.decision_function(self.scaler.transform(X))


class OCSVM(_Base):
    def __init__(self, nu=0.05, max_train=2500, seed=42):
        self.scaler = StandardScaler()
        self.model = OneClassSVM(nu=nu, gamma="scale")
        self.max_train, self.seed = max_train, seed

    def fit(self, X):
        A = self.scaler.fit_transform(X)
        rng = np.random.default_rng(self.seed)
        sub = A[rng.choice(len(A), min(self.max_train, len(A)), replace=False)]
        self.model.fit(sub)
        self.train_scores_ = -self.model.decision_function(A)
        return self

    def score(self, X):
        return -self.model.decision_function(self.scaler.transform(X))


class RankEnsemble(_Base):
    """Har model ka score 0-1 'percentile' mein badal kar average. Alag alag scales
    wale models ko seedha jodna galat hota hai, isliye percentile use hota hai."""

    def __init__(self, models):
        self.models = models

    def fit(self, X):
        for m in self.models:
            m.fit(X)
        self._sorted = [np.sort(m.train_scores_) for m in self.models]
        self.train_scores_ = self._combine([m.train_scores_ for m in self.models])
        return self

    def _combine(self, score_list):
        ranks = [np.searchsorted(s, sc) / len(s) for s, sc in zip(self._sorted, score_list)]
        return np.mean(ranks, axis=0)

    def score(self, X):
        return self._combine([m.score(X) for m in self.models])
