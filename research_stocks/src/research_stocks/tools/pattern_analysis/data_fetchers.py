# data_fetchers.py
# ---------------
# Functions for fetching stock market data from various sources

import os
from datetime import datetime, time, timedelta

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


def fetch_hourly_data(symbol: str, days: int = 20) -> pd.DataFrame:
  """Fetch hourly OHLC data for the specified symbol for the last N days."""
  end_date = datetime.now()
  start_date = end_date - timedelta(days=days)

  df_hourly = yf.Ticker(symbol).history(
      start=start_date.strftime("%Y-%m-%d"),
      end=end_date.strftime("%Y-%m-%d"),
      interval="1h")

  df_hourly = df_hourly.reset_index()
  df_hourly["Date"] = df_hourly["Date"].dt.strftime("%Y-%m-%d %H:%M")
  return df_hourly


def fetch_minutes_data(symbol: str, interval: int = 15, days: int = 10) -> pd.DataFrame:
  """Fetch N-minute OHLC data for the specified symbol for the last M days."""
  valid_intervals = {1: "1m", 2: "2m", 5: "5m", 15: "15m", 30: "30m", 60: "60m", 90: "90m"}
  yf_interval = valid_intervals.get(interval, "15m")

  max_days = 7 if interval == 1 else 60
  fetch_days = min(days, max_days)

  end_date = datetime.now()
  start_date = end_date - timedelta(days=fetch_days)

  df_minutes = yf.Ticker(symbol).history(
      start=start_date.strftime("%Y-%m-%d"),
      end=end_date.strftime("%Y-%m-%d"),
      interval=yf_interval)

  df_minutes = df_minutes.reset_index()
  df_minutes["Date"] = df_minutes["Date"].dt.strftime("%Y-%m-%d %H:%M")
  df_minutes = df_minutes[df_minutes["Date"].str.split(" ").str[1].between("09:30", "16:00")]
  return df_minutes


def fetch_polygon_intraday(symbol: str, api_key: str, interval: int = 15,
    days: int = 10, limit: int = 1000) -> pd.DataFrame:
  """Alternative implementation using Polygon.io for intraday data."""
  end_date = datetime.now()
  start_date = end_date - timedelta(days=days)

  url = (f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{interval}/minute/"
         f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
         f"?adjusted=true&sort=asc&limit={limit}&apiKey={api_key}")

  try:
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    if not data.get("results"):
      print(f"⚠️ No {interval}-minute data returned for the specified period.")
      return pd.DataFrame()

    bars = [
      {"Datetime": datetime.fromtimestamp(bar["t"] / 1000),
       "Open": bar["o"], "High": bar["h"], "Low": bar["l"],
       "Close": bar["c"], "Volume": bar.get("v", 0)}
      for bar in data["results"]
    ]
    df = pd.DataFrame(bars)

    df = df[df["Datetime"].dt.time.between(time(9, 30), time(16, 0))]

    if "Date" not in df.columns:
      df["Date"] = df["Datetime"]
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d %H:%M")

    return df

  except Exception as exc:
    print(f"❌ Error fetching {interval}-minute bars: {exc}")
    return pd.DataFrame()
