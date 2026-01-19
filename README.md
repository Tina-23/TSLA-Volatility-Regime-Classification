# 📈 TSLA Volatility Regime Classification

## Project Overview

This project builds a **machine learning system to classify Tesla (TSLA) stock trading days into volatility regimes** — **Low**, **Medium**, and **High volatility** — using historical price and volume data.

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

- **Source**: Kaggle – TSLA historical stock prices  
- **Frequency**: Daily  
- **Raw columns**:
  - `Date`
  - `Open`
  - `High`
  - `Low`
  - `Close`
  - `Volume`

The raw dataset contains **no missing values**.

---

## Data Cleaning

The following preprocessing steps were applied:
- Removed redundant index column (`Unnamed: 0`)
- Converted `Date` to datetime format
- Sorted observations chronologically
- Dropped rows created by rolling window feature calculations

---

## Feature Engineering

The following features were engineered from raw price and volume data:

| Feature | Description |
|------|------------|
| `return` | Daily percentage return |
| `vol_10` | 10-day rolling volatility of returns |
| `vol_20` | 20-day rolling volatility of returns |
| `price_range` | Normalized daily range `(High − Low) / Close` |
| `volume_change` | Daily percentage change in trading volume |

These features are standard in financial modeling and are both **interpretable and robust**.

---

## Target Variable: Volatility Regimes

The target variable is a **three-class volatility regime label**, derived from the 20-day rolling volatility.

| Label | Regime |
|----|----|
| 0 | Low volatility |
| 1 | Medium volatility |
| 2 | High volatility |

Regimes are defined using **quantile-based thresholds**:
- Bottom 33% → Low
- Middle 33% → Medium
- Top 33% → High

This avoids arbitrary cutoffs and adapts naturally to TSLA’s volatility distribution.

---

## Exploratory Data Analysis (EDA)

Key insights from EDA:
- Volatility clusters over time rather than appearing randomly
- High-volatility regimes align with sharp price movements
- Volume changes increase during volatile periods

Visualizations included:
- Rolling volatility over time
- Distribution of volatility regimes
- TSLA closing price colored by regime
- Feature correlation heatmap

---

## Modeling Approach

### Train–Test Split
- **Time-aware split** (80% train, 20% test)
- No random shuffling to avoid look-ahead bias

### Baseline Model
- Logistic Regression (with feature scaling)

### Final Model
- **Random Forest Classifier**
- Implemented using a **scikit-learn Pipeline**
- Hyperparameters tuned conservatively to reduce overfitting

### Evaluation Metrics
- Precision, Recall, and F1-score (macro-averaged)
- Confusion matrix

The Random Forest model outperformed the baseline and was selected for deployment.

---

## Model Deployment (FastAPI)

The trained pipeline is deployed using **FastAPI**, providing a REST API for volatility regime predictions.

## Reproducibility — Quick start

These steps reproduce training and the API locally.

1. Create virtual environment and install dependencies:
   - python -m venv venv
   - source venv/bin/activate  # or venv\Scripts\activate on Windows
   - pip install -r requirements.txt

2. Download the data (the repository expects `TSLA_Stock.csv`):
   - python download_data.py --start 2015-01-01 --end 2024-12-31 --out TSLA_Stock.csv

3. Train the model (saves pipeline to `model/volatility_model.pkl` and metrics to `model/metrics.json`):
   - python train.py --data TSLA_Stock.csv --model-dir model

4. Run the API locally:
   - uvicorn app:app --reload --host 0.0.0.0 --port 8000
   - POST /predict with JSON matching the `MarketFeatures` schema in `app.py`.

Notes:
- If you prefer Docker:
  - docker build -t tsla-vol-regime .
  - docker run -p 8000:8000 tsla-vol-regime
- The train script uses a time-aware split and a light randomized search to tune the RandomForest. Check `model/metrics.json` for evaluation results and the split dates used.


