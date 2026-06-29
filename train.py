
import argparse
import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import classification_report, balanced_accuracy_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    df["return"]        = df["Close"].pct_change()
    df["vol_5"]         = df["return"].rolling(5).std()
    df["vol_10"]        = df["return"].rolling(10).std()
    df["vol_20"]        = df["return"].rolling(20).std()
    df["price_range"]   = (df["High"] - df["Low"]) / df["Close"]
    df["volume_change"] = df["Volume"].pct_change()
    df["return_5d"]     = df["Close"].pct_change(5)

    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

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

    features = ["return", "vol_5", "vol_10", "vol_20", "price_range", "volume_change", "return_5d", "rsi_14"]
    X = df[features]
    y = df["volatility_regime"]

    split_idx = int(len(df) * (1.0 - args.test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    tscv = TimeSeriesSplit(n_splits=3)

    # --- Random Forest (hyperparameter search) ---
    rf_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(class_weight="balanced", random_state=args.random_state))
    ])
    rf_param_dist = {
        "model__n_estimators": randint(100, 401),
        "model__max_depth": [None, 5, 8, 10, 12],
        "model__min_samples_split": randint(2, 11),
    }
    rf_search = RandomizedSearchCV(
        rf_pipeline, rf_param_dist, n_iter=12, cv=tscv,
        scoring="f1_macro", random_state=args.random_state, n_jobs=args.n_jobs, verbose=1,
    )
    rf_search.fit(X_train, y_train)
    rf_best = rf_search.best_estimator_
    y_pred_rf = rf_best.predict(X_test)

    # --- Gradient Boosting (fixed params) ---
    gb_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            subsample=0.8, random_state=args.random_state
        ))
    ])
    gb_pipeline.fit(X_train, y_train)
    y_pred_gb = gb_pipeline.predict(X_test)

    # Pick the model with better balanced accuracy
    bal_rf = balanced_accuracy_score(y_test, y_pred_rf)
    bal_gb = balanced_accuracy_score(y_test, y_pred_gb)
    if bal_rf >= bal_gb:
        best, y_pred, chosen = rf_best, y_pred_rf, "random_forest"
        bal_acc = bal_rf
    else:
        best, y_pred, chosen = gb_pipeline, y_pred_gb, "gradient_boosting"
        bal_acc = bal_gb

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "volatility_model.pkl")
    joblib.dump(best, model_path)

    metrics = {
        "chosen_model": chosen,
        "balanced_accuracy": float(bal_acc),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "rf_balanced_accuracy": float(bal_rf),
        "gb_balanced_accuracy": float(bal_gb),
        "classification_report": report,
        "best_rf_params": rf_search.best_params_,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_end_date": str(df.loc[split_idx - 1, "Date"]) if split_idx > 0 else None,
        "test_start_date": str(df.loc[split_idx, "Date"]) if split_idx < len(df) else None,
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
