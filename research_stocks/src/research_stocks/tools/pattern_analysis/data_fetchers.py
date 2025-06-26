# data_fetchers.py
# ---------------
# Functions for fetching stock market data from various sources

import os
from datetime import datetime, time

import pandas as pd
import requests
import yfinance as yf


def fetch_intraday_bars(symbol: str, api_key: str,
    limit: int = 150) -> pd.DataFrame | None:
  """
  Pulls the latest `limit` 1-minute bars for `symbol` from Polygon.io.
  Returns a DataFrame or None if no data.
  """
  today = datetime.now().strftime("%Y-%m-%d")
  url = (f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/"
         f"{today}/{today}?adjusted=true&sort=asc&limit={limit}&apiKey={api_key}")

  try:
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if not data.get("results"):
      print("⚠️  No intraday data returned (market closed or key expired).")
      return None

    bars = [
      {"Datetime": datetime.fromtimestamp(bar["t"] / 1000), "Open": bar["o"],
       "High": bar["h"], "Low": bar["l"], "Close": bar["c"],
       "Volume": bar.get("v", 0), } for bar in data["results"]]
    df = pd.DataFrame(bars)

    # ► Keep only regular-hours bars (09:30–16:00 ET). Comment out to include pre-/post-market.
    df = df[df["Datetime"].dt.time.between(time(9, 30), time(16, 0))]

    # ── Harmonise column names for downstream helpers ──
    # Many pattern‑detection utilities expect a 'Date' field identical
    # to the daily‑candle DataFrames.  Keep both columns so nothing else breaks.
    if "Date" not in df.columns:
      df["Date"] = df["Datetime"]
    # Store as string like the daily frame, e.g. '2025-06-26 11:03'
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d %H:%M")

    return df

  except Exception as exc:
    print(f"❌ Error fetching intraday bars: {exc}")
    return None


def fetch_daily_history(symbol: str, period: str = "12mo") -> pd.DataFrame:
  """
  Fetches daily historical data for the given symbol using yfinance.
  
  Args:
      symbol: The stock symbol to fetch data for
      period: Time period to fetch (e.g., "12mo", "1y", "max")
      
  Returns:
      DataFrame with daily OHLCV data
  """
  ticker = yf.Ticker(symbol)
  df_hist = ticker.history(period=period, interval="1d").iloc[:-1].copy()
  df_hist.reset_index(inplace=True)
  df_hist["Date"] = df_hist["Date"].dt.strftime("%Y-%m-%d")
  
  return df_hist