
import argparse
import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    df["return"] = df["Close"].pct_change()
    df["vol_10"] = df["return"].rolling(10).std()
    df["vol_20"] = df["return"].rolling(20).std()
    df["price_range"] = (df["High"] - df["Low"]) / df["Close"]
    df["volume_change"] = df["Volume"].pct_change()

    vol = df["vol_20"]
    low_q = vol.quantile(0.33)
    high_q = vol.quantile(0.66)

    def assign_regime(v):
        if v <= low_q:
            return 0
        elif v <= high_q:
            return 1
        else:
            return 2

    df["volatility_regime"] = vol.apply(assign_regime)
    df = df.dropna().reset_index(drop=True)
    return df

def train(args):
    df = pd.read_csv(args.data)
    df = compute_features(df)

    features = ["return", "vol_10", "vol_20", "price_range", "volume_change"]
    X = df[features]
    y = df["volatility_regime"]

    split_idx = int(len(df) * (1.0 - args.test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(random_state=args.random_state))
    ])

    # Light hyperparameter search with time-series splits
    tscv = TimeSeriesSplit(n_splits=3)
    param_dist = {
        "model__n_estimators": randint(50, 401),
        "model__max_depth": [None, 5, 8, 12],
        "model__min_samples_split": randint(2, 11)
    }

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=12,
        cv=tscv,
        scoring="f1_macro",
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        verbose=1,
    )

    search.fit(X_train, y_train)

    best = search.best_estimator_
    y_pred = best.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    bal_acc = balanced_accuracy_score(y_test, y_pred)

    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "volatility_model.pkl")
    joblib.dump(best, model_path)

    metrics = {
        "balanced_accuracy": float(bal_acc),
        "classification_report": report,
        "best_params": search.best_params_,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_end_date": str(df.loc[split_idx - 1, "Date"]) if split_idx > 0 else None,
        "test_start_date": str(df.loc[split_idx, "Date"]) if split_idx < len(df) else None
    }

    metrics_path = os.path.join(args.model_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model to {model_path}")
    print(f"Saved metrics to {metrics_path}")
    return model_path, metrics_path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to TSLA_Stock.csv")
    parser.add_argument("--model-dir", default="model", help="Directory to save trained model and metrics")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proportion for the test set (time series split)")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
