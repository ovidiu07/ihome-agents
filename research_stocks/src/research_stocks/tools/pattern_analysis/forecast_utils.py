# forecast_utils.py
# ----------------
# Functions for forecasting and bias calculation

import numpy as np
import pandas as pd


def get_intraday_bias(patterns: list[dict]) -> int:
  """Return +1 for net bullish, ‑1 for net bearish, 0 otherwise."""
  if not patterns:
    return 0
  bulls = sum(p["direction"] == "bullish" for p in patterns)
  bears = sum(p["direction"] == "bearish" for p in patterns)
  return (bulls > bears) - (bears > bulls)


def get_daily_bias(patterns: list[dict]) -> int:
  """Same as above but for daily frame."""
  return get_intraday_bias(patterns)


def blended_forecast(intraday_direction: int, daily_direction: int,
    vwap_trend: str, atr: float) -> str:
  """
  Very simple ensemble: majority vote of three signals.
  Returns 'UP', 'DOWN' or 'NEUTRAL'.
  """
  votes = [intraday_direction, daily_direction, 1 if vwap_trend == "UP" else -1]
  score = sum(votes)
  if score > 1:
    return f"UP (target +{atr:.2f})"
  if score < -1:
    return f"DOWN (target -{atr:.2f})"
  return "NEUTRAL"


def calculate_vwap_obv_trend(df: pd.DataFrame) -> str:
  """
  Calculate VWAP and OBV trend indicators.
  
  Args:
      df: DataFrame with OHLCV data
      
  Returns:
      String indicating trend direction ("UP" or "DOWN")
  """
  if "Volume" not in df.columns or df["Volume"].isnull().all():
    return "NEUTRAL"
    
  # VWAP — stays a Series
  vwap = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()
  
  # OBV — force a Series (np.where returns ndarray)
  obv_raw = np.where(df["Close"].diff().fillna(0) >= 0, df["Volume"], -df["Volume"]).cumsum()
  obv = pd.Series(obv_raw, index=df.index)
  
  # Determine trend
  trend = "UP" if (df["Close"].iloc[-1] > vwap.iloc[-1] and obv.iloc[-1] > obv.iloc[0]) else "DOWN"
  
  return trend


def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
  """
  Calculate Average True Range (ATR) for volatility estimation.
  
  Args:
      df: DataFrame with OHLC data
      period: Period for ATR calculation
      
  Returns:
      ATR value
  """
  return df["High"].sub(df["Low"]).rolling(period).mean().iloc[-1]