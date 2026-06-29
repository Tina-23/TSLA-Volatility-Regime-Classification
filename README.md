# TSLA Volatility Regime Classification

## Project Overview

This project builds a **supervised machine learning system to classify Tesla (TSLA) stock trading days into volatility regimes** — **Low**, **Medium**, and **High** — using historical price and volume data.

Rather than predicting future stock prices (which is noisy and unreliable), the project focuses on **volatility regime classification**, a task commonly used in **risk management, portfolio allocation, and market monitoring**.

The final trained model is deployed using **FastAPI**, exposing a REST endpoint for real-time predictions.

---

## Problem Statement

**Can we classify TSLA trading days into different volatility regimes using engineered features derived from historical price and volume data?**

### Why volatility regimes?
- Volatility reflects **market uncertainty and risk**
- Regime-based modeling is more stable than price prediction
- Widely used in quantitative finance and risk analytics

---

## Dataset

- **Source**: Yahoo Finance via `yfinance` (2015–2024)
- **Frequency**: Daily OHLCV
- **Size**: ~2,496 trading days after feature engineering
- **Raw columns**: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`

---

## Data Cleaning

- Converted `Date` to datetime and sorted chronologically
- Dropped rows with NaN values introduced by rolling window calculations (~20 rows)

---

## Feature Engineering

Eight features were engineered from raw price and volume data:

| Feature | Description |
|---|---|
| `return` | Daily percentage return |
| `vol_5` | 5-day rolling volatility of returns |
| `vol_10` | 10-day rolling volatility of returns |
| `vol_20` | 20-day rolling volatility (used to define regimes) |
| `price_range` | Normalized daily range `(High − Low) / Close` |
| `volume_change` | Daily percentage change in trading volume |
| `return_5d` | 5-day cumulative return (momentum) |
| `rsi_14` | 14-day Relative Strength Index |

---

## Target Variable: Volatility Regimes

The target is a **three-class label** derived from the 20-day rolling volatility using quantile-based thresholds:

| Label | Regime | Threshold |
|---|---|---|
| 0 | Low | vol_20 ≤ 33rd percentile |
| 1 | Medium | 33rd < vol_20 ≤ 66th percentile |
| 2 | High | vol_20 > 66th percentile |

This produces a near-balanced class distribution (~33% each) that adapts naturally to TSLA's volatility profile over the full data range.

---

## Exploratory Data Analysis

- Volatility clusters over time rather than appearing randomly
- High-volatility regimes align with sharp price movements (e.g. COVID-19, 2022 rate hikes)
- Volume changes increase during volatile periods

Visualizations:
- 20-day rolling volatility with Low/Medium/High threshold lines
- Regime distribution bar chart
- TSLA closing price colored by regime
- Feature importance comparison chart

---

## Modeling

### Train–Test Split
- **Time-aware split**: 80% train (2015–2023), 20% test (2023–2024)
- No random shuffling to prevent look-ahead bias
- All three regimes are represented in both splits

### Models Trained

| Model | Role |
|---|---|
| Logistic Regression | Baseline |
| Random Forest | Primary classifier (with RandomizedSearchCV) |
| Gradient Boosting | Primary classifier (n_estimators=300, lr=0.05) |

Both Random Forest and Gradient Boosting use scikit-learn `Pipeline` objects (scaler → model).

### Results on Held-Out Test Set (500 samples)

| Model | Accuracy | Balanced Accuracy |
|---|---|---|
| Logistic Regression | 97.2% | 96.9% |
| Random Forest | 99.8% | 99.8% |
| Gradient Boosting | 99.8% | 99.8% |

Both ensemble models achieve **~85%+ classification accuracy** on held-out test data, comfortably exceeding the project target.

### Evaluation Metrics
- Per-class precision, recall, and F1-score
- Balanced accuracy (accounts for class imbalance)
- Confusion matrices for RF and Gradient Boosting

---

## Model Deployment (FastAPI)

The best model pipeline is saved to `model/volatility_model.pkl` and served via FastAPI.

**Endpoint**: `POST /predict`

```json
{
  "return_": -0.015,
  "vol_10": 0.022,
  "vol_20": 0.018,
  "price_range": 0.031,
  "volume_change": 0.12
}
```

**Response**:
```json
{
  "volatility_regime": 0,
  "description": "Low Volatility"
}
```

---

## Reproducibility — Quick Start

1. **Install dependencies**
   ```
   pip install -r Requirement.txt
   ```

2. **Download data** (saves `TSLA_Stock.csv`)
   ```
   python download_dataa.py --start 2015-01-01 --end 2025-01-01 --out TSLA_Stock.csv
   ```

3. **Train models** (saves pipeline and metrics to `model/`)
   ```
   python train.py --data TSLA_Stock.csv --model-dir model
   ```

4. **Run the API**
   ```
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

5. **Or use Docker**
   ```
   docker build -t tsla-vol-regime .
   docker run -p 8000:8000 tsla-vol-regime
   ```

Check `model/metrics.json` for evaluation results and split dates after training.
