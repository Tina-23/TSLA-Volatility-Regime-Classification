train.py
import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load data
df = pd.read_csv("TSLA_Stock.csv")

# Cleaning
df = df.drop(columns=["Unnamed: 0"])
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

# Feature engineering
df["return"] = df["Close"].pct_change()
df["vol_10"] = df["return"].rolling(10).std()
df["vol_20"] = df["return"].rolling(20).std()
df["price_range"] = (df["High"] - df["Low"]) / df["Close"]
df["volume_change"] = df["Volume"].pct_change()

# Target
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

df = df.dropna()

features = [
    "return",
    "vol_10",
    "vol_20",
    "price_range",
    "volume_change"
]

X = df[features]
y = df["volatility_regime"]

# Train-test split (time-based)
split = int(len(df) * 0.8)
X_train, y_train = X.iloc[:split], y.iloc[:split]

# Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42
    ))
])

pipeline.fit(X_train, y_train)

# Save model
joblib.dump(pipeline, "model/volatility_model.pkl")

print("Model saved successfully")
