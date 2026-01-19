"""
download_data.py

Usage:
    python download_data.py --start 2015-01-01 --end 2025-01-01 --out TSLA_Stock.csv

This script downloads historical TSLA daily OHLCV data via yfinance
and writes a CSV with columns: Date, Open, High, Low, Close, Volume.
"""
import argparse
from datetime import date
import yfinance as yf
import pandas as pd

def download(symbol: str, start: str, end: str, out_path: str):
    df = yf.download(symbol, start=start, end=end, progress=False)
    if df.empty:
        raise RuntimeError("No data downloaded. Check dates or network.")
    df = df.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="TSLA")
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default=str(date.today()))
    parser.add_argument("--out", type=str, default="TSLA_Stock.csv")
    args = parser.parse_args()
    download(args.symbol, args.start, args.end, args.out)

if __name__ == "__main__":
    main()
