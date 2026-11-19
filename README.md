# ANOMIVUE Intelligence

**Anomaly + View.** A dashboard and model for finding and investigating unusual social media engagement patterns (viral spikes, bot-like post bursts, sudden drops).

> **Read this first.** The dataset in this repository has no ground-truth anomaly labels and looks simulated (likes, shares and impressions are uncorrelated, hashtags do not match post text). Results below come from anomalies that were **injected on purpose** into a held-out test set. They show that the pipeline works. They do **not** show how well it would work on real platform data.

---

## What it does

- Scores each platform-hashtag record with a **Local Outlier Factor (LOF)** model trained on log-scaled engagement features.
- Flags the most unusual 5% and explains each flag in plain words (for example, "Post volume high (4 vs typical 1)").
- Shows the results in a Next.js dashboard: overview, anomaly table, trends, insights, data explorer, model performance, and a live simulator.
- The simulator runs the **real model** in TypeScript (no Python server needed), checked against the Python model to about 1e-6.

## How it was evaluated

The labels do not exist, so test anomalies were created:

1. Train on the first 70% of dates. Test on the last 30%.
2. Alter 5% of test rows in three ways: **viral spike** (likes and shares 5-10x), **bot burst** (6-12 posts in one window), **engagement crash** (likes, shares and engagement near zero).
3. Repeat with 5 random seeds and average.

| Model | PR-AUC | Best F1 | Recall (5% threshold) |
| --- | --- | --- | --- |
| Isolation Forest (raw features) | 0.355 | 0.501 | 0.620 |
| Z-score rule (raw features) | 0.702 | 0.718 | 0.662 |
| LOF (raw features) | 0.782 | 0.793 | 0.728 |
| Isolation Forest (log features) | 0.544 | 0.595 | 0.738 |
| One-Class SVM (log features) | 0.929 | 0.866 | 0.991 |
| **LOF (log features), used here** | **0.972** | **0.928** | **0.990** |
| Average of IF + LOF + SVM (log) | 0.797 | 0.737 | 0.975 |
| LOF (log + time features) | 0.938 | 0.861 | 0.974 |

What this says:

- The biggest gain came from the **features** (taking the log of likes, shares and engagement), not from the algorithm.
- The first version of this project used Isolation Forest on raw numbers. It scored **below a simple z-score rule**.
- Averaging models did not help, and time-based features did not help (this data has no real time pattern).

Full tables: `results/model_comparison.csv`, `results/model_recall_by_type.csv`, `results/metrics.csv`.

## Limitations

- **Synthetic-looking data.** Real anomalies (coordinated campaigns, odd combinations) may not look like the injected ones.
- **Circular test.** Injected anomalies are multiplicative ("x times bigger or smaller"), which log features are good at. The feature choice was also made by looking at the same benchmark, so the numbers are probably optimistic.
- **The 5% contamination is an assumption**, not a measured rate. Because 5% were injected in testing, the evaluation cannot confirm that 5% is right.
- **About 31% of flagged records have no single extreme metric.** LOF flagged them because the combination is rare. They are harder to explain and should be reviewed by hand.
- Precision at the 5% threshold is about 0.49: the threshold flags roughly 5% of normal rows too.
- The login screen is a demo. Credentials are in the browser code. Do not treat it as security.
- Historical data only. There is no live platform ingestion.

## Quick start

Requirements: Python 3.11+, Node 18+.

```bash
# 1. Python side: train the model and regenerate dashboard data
pip install -r requirements.txt
python train_model.py        # writes anomivue_model.pkl, anomaly_output.csv, src/data/*.json

# 2. Evaluation (optional, about 15 seconds each)
python evaluate.py           # Isolation Forest vs z-score on injected anomalies
python compare_models.py     # all models, time-based split

# 3. Web app
npm install
npm run dev                  # http://localhost:3000

# 4. Python API (optional)
uvicorn backend.index:app --reload --port 8000
```

Demo login: `admin` / `1234` (see note above).

## Project structure

```
config.py            settings in one place (contamination, k, paths)
models.py            features + models (z-score, Isolation Forest, LOF, One-Class SVM, ensemble)
scoring.py           production LOF: training, scoring, severity, explanations, exports
train_model.py       one command: train + export everything
evaluate.py          injected-anomaly test (Isolation Forest vs z-score)
compare_models.py    full model comparison
backend/index.py     FastAPI version of the API
src/lib/lof.ts       TypeScript LOF (same model, runs on Vercel)
src/lib/insights.ts  insights calculated from the data
src/components/      dashboard views
src/app/api/         Next.js API routes
results/             evaluation tables
```

## How scoring works

- **LOF** compares how crowded a record's neighbourhood is (its 35 nearest training records) with how alone the record is. About 1 is typical. Above the threshold (1.2327 on this data) is flagged.
- **Severity** (0-100): `100 * (1 - 2^(-(LOF - 1) / (threshold - 1)))`. The flag boundary is 50.
- **Features:** `post_count`, `log(1 + likes)`, `log(1 + shares)`, `log(1 + engagement_rate)`.

## API (Next.js routes)

| Route | Purpose |
| --- | --- |
| `POST /api/predict` | Score one record with the real model |
| `GET /api/model/info` | Model settings and counts |
| `GET /api/insights?platform=` | Insights computed from the data |
| `GET /api/health` | Status |

Example:

```bash
curl -X POST localhost:3000/api/predict -H "Content-Type: application/json" \
  -d '{"post_count":3,"likes":14500,"shares":5100,"engagement_rate":1.85}'
```

```json
{
  "is_anomaly": true,
  "anomaly_score": 99.9,
  "lof_score": 3.304,
  "threshold": 1.2327,
  "explanation": "Post volume high (3 vs typical 1); Engagement rate unusually high (22x typical)"
}
```

(Other fields are in the real response.)

## Roadmap

- [ ] Test on real labelled data (the most important next step)
- [ ] Add anomaly types that are not simply "x times bigger or smaller"
- [ ] Use the sentiment and toxicity columns from the raw data
- [ ] Apply the same idea to real comments collected from a platform
- [ ] Alerts (email or chat)

## Author

**Hoor-Ain Rajput** - [github.com/hoorainrajput](https://github.com/hoorainrajput)

Author of this version (ANOMIVUE Intelligence): the new name and theme, the LOF model on log features, the injected-anomaly evaluation (`evaluate.py`, `compare_models.py`), the TypeScript model used by the simulator, data-driven insights, and this documentation.

**License: not specified.** Before publishing or reusing this repository, confirm permission with the original author or replace the inherited parts.