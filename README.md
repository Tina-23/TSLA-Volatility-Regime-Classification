# TSLA-Volatility-Regime-Classification
This project builds a machine learning system to classify Tesla (TSLA) stock trading days into volatility regimes — Low, Medium, and High volatility — using historical price and volume data.

Rather than predicting future stock prices (which is noisy and unreliable), the project focuses on volatility regime classification, a task that is widely used in risk management, portfolio allocation, and market monitoring.

The final trained model is deployed using FastAPI, exposing a REST endpoint for real-time predictions.

Problem Statement

Can we classify TSLA trading days into different volatility regimes using engineered features derived from historical price and volume data?

Why volatility regimes?

Volatility reflects market uncertainty and risk

Regime-based analysis is more stable than price prediction

Commonly used in quantitative finance and risk analytics

Dataset

Source: Kaggle – TSLA historical stock prices

Frequency: Daily

Columns used:

Date

Open

High

Low

Close

Volume

The dataset contains no missing values in raw form.

Data Cleaning

Removed redundant index column (Unnamed: 0)

Converted Date to datetime format

Sorted data chronologically

Dropped rows introduced by rolling window calculations

Feature Engineering

The following features were engineered from raw price and volume data:

Feature	Description
return	Daily percentage return
vol_10	10-day rolling volatility of returns
vol_20	20-day rolling volatility of returns
price_range	Normalized daily price range (High − Low) / Close
volume_change	Daily percentage change in trading volume

These features are standard in financial modeling and are interpretable and robust.

Target Variable: Volatility Regimes

The target variable is a three-class volatility regime label, created using the 20-day rolling volatility:

Label	Regime
0	Low volatility
1	Medium volatility
2	High volatility

Regimes are defined using quantile-based thresholds:

Bottom 33% → Low

Middle 33% → Medium

Top 33% → High

This avoids arbitrary cutoffs and adapts naturally to TSLA’s volatility distribution.

Exploratory Data Analysis (EDA)

Key EDA insights include:

Volatility clusters over time rather than being randomly distributed

High-volatility regimes often coincide with sharp price movements

Volume changes correlate positively with volatility spikes

Visualizations included:

Rolling volatility over time

Distribution of volatility regimes

TSLA price colored by regime

Feature correlation analysis

Modeling Approach
Train–Test Split

Time-aware split (80% train, 20% test)

No random shuffling to avoid look-ahead bias

Baseline Model

Logistic Regression (with feature scaling)

Final Model

Random Forest Classifier

Implemented using a scikit-learn Pipeline

Hyperparameters tuned conservatively to avoid overfitting

Evaluation Metrics

Precision, Recall, F1-score (macro averaged)

Confusion matrix

The Random Forest model outperformed the baseline and was selected for deployment.

Model Deployment (FastAPI)

The trained pipeline is deployed using FastAPI, providing a REST API for volatility regime predictions.

API Endpoints
Health Check
GET /


Response:

{ "status": "ok" }

Prediction
POST /predict


Input

{
  "return_": 0.012,
  "vol_10": 0.028,
  "vol_20": 0.035,
  "price_range": 0.045,
  "volume_change": 0.12
}


Output

{
  "volatility_regime": 2,
  "description": "High Volatility"
}

API Features

Pydantic schema validation

Human-readable regime labels

Auto-generated Swagger UI (/docs)

Project Structure
project/
│
├── model/
│   └── volatility_model.pkl
│
├── train.py
├── app.py
├── requirements.txt
└── README.md

Technologies Used

Python

pandas, numpy

scikit-learn

matplotlib

FastAPI

Uvicorn

joblib

How to Run the Project
1. Install Dependencies
pip install -r requirements.txt

2. Train the Model
python train.py

3. Start the API
uvicorn app:app --reload


Visit:

http://127.0.0.1:8000/docs

Limitations

The model is trained on a single stock (TSLA) and may not generalize to other assets

No macroeconomic or market-wide indicators are included

Volatility regimes are based on historical rolling windows and do not imply future certainty

The project uses historical simulation only and does not include live data ingestion

Future Improvements

Extend to multiple stocks or indices

Incorporate macroeconomic indicators

Add LSTM-based temporal models

Perform anomaly detection as a complementary task

Explore quantum feature maps as a research extension

Author Notes

This project was developed as a final capstone for the Machine Learning Zoomcamp, emphasizing:

Clean data pipelines

Proper problem framing

Reproducibility

Practical deployment
