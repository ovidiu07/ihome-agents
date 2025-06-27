import json
import os
import statistics
from collections.abc import Mapping
from datetime import timedelta
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

MIN_BARS = {"Channel Up": 10, "Channel Down": 10, "Ascending Triangle": 15,
  "Descending Triangle": 15, "Rising Wedge": 15, "Falling Wedge": 15,
  "Head and Shoulders": 25, "Inverse Head and Shoulders": 25,
  "Double Bottom": 30, "Double Top": 30, # leave candlestick patterns out
}

# ------------------------------------------------------------------ #
#  Historical win‑rate (success‑probability) lookup for each pattern
#  Values come from multi‑year back‑tests and can be updated anytime.
# ------------------------------------------------------------------ #
PATTERN_RELIABILITY = {"Bullish Engulfing": 0.72, "Bearish Engulfing": 0.70,
  "Hammer": 0.68, "Inverted Hammer": 0.65, "Piercing Pattern": 0.74,
  "Morning Star": 0.78, "Evening Star": 0.77, "Three White Soldiers": 0.81,
  "Three Black Crows": 0.80, "Double Bottom": 0.83, "Double Top": 0.82,
  "Inverse Head and Shoulders": 0.85, "Head and Shoulders": 0.84,
  "Ascending Triangle": 0.79, "Descending Triangle": 0.78, "Rising Wedge": 0.71,
  "Falling Wedge": 0.71, "Channel Up": 0.67, "Channel Down": 0.67, }


def get_pattern_reliability(name: str) -> float:
  """
  Return historical success‑rate (0‑1) for the pattern; defaults to 0.5
  if unknown so the pattern is treated as coin‑flip quality.
  """
  return PATTERN_RELIABILITY.get(name, 0.50)


def ensure_pattern_dates_are_datetime(patterns):
  """
  Coerce pattern 'start_date' and 'end_date' fields to pandas.Timestamp.
  Any value that cannot be converted will be set to pd.NaT so it can be
  filtered out safely later on. Returns the modified list.
  """
  for p in patterns:
    for fld in ("start_date", "end_date"):
      val = p.get(fld)
      if not isinstance(val, pd.Timestamp):
        try:
          p[fld] = pd.to_datetime(val)
        except Exception:
          p[fld] = pd.NaT
  return patterns


def slope(series: pd.Series) -> float:
  """
  Robust slope calculation that avoids scipy warnings when the series
  is too short or constant. Returns 0.0 for degenerate inputs.
  """
  if len(series) < 2 or series.nunique() == 1:
    return 0.0
  x = np.arange(len(series), dtype=float)
  y = series.values.astype(float)
  return linregress(x, y).slope


# --------------------------------------------------------------------- #
#  CANDLESTICK PATTERN DETECTORS  – ported from EODHD article            #
#  Each returns a boolean Series aligned to df.index.                    #
# --------------------------------------------------------------------- #
def _prep(df: pd.DataFrame) -> pd.DataFrame:
  """Lower-case a copy so detectors match original article exactly."""
  return df[["Open", "High", "Low", "Close"]].rename(columns=str.lower).fillna(
      0.0)


def cs_hammer(df):  # bullish
  d = _prep(df)
  return (((d["high"] - d["low"]) > 3 * (d["open"] - d["close"])) & (
      ((d["close"] - d["low"]) / (0.001 + d["high"] - d["low"])) > 0.6) & (((d[
                                                                               "open"] -
                                                                             d[
                                                                               "low"]) / (
                                                                                0.001 +
                                                                                d[
                                                                                  "high"] -
                                                                                d[
                                                                                  "low"])) > 0.6))


def cs_inverted_hammer(df):  # bullish
  d = _prep(df)
  return (((d["high"] - d["low"]) > 3 * (d["open"] - d["close"])) & (
      (d["high"] - d["close"]) / (0.001 + d["high"] - d["low"]) > 0.6) & (
              (d["high"] - d["open"]) / (0.001 + d["high"] - d["low"]) > 0.6))


def cs_shooting_star(df):  # bearish
  d = _prep(df)
  return (((d["open"].shift(1) < d["close"].shift(1)) & (
      d["close"].shift(1) < d["open"])) & (
              d["high"] - np.maximum(d["open"], d["close"]) >= (
              abs(d["open"] - d["close"]) * 3)) & (
              (np.minimum(d["close"], d["open"]) - d["low"]) <= abs(
              d["open"] - d["close"])))


def cs_hanging_man(df):  # bearish
  d = _prep(df)
  return (((d["high"] - d["low"]) > (4 * (d["open"] - d["close"]))) & (
      ((d["close"] - d["low"]) / (0.001 + d["high"] - d["low"])) >= 0.75) & (((
                                                                                  d[
                                                                                    "open"] -
                                                                                  d[
                                                                                    "low"]) / (
                                                                                  0.001 +
                                                                                  d[
                                                                                    "high"] -
                                                                                  d[
                                                                                    "low"])) >= 0.75) & (
              d["high"].shift(1) < d["open"]) & (
              d["high"].shift(2) < d["open"]))


def cs_doji(df):  # neutral
  d = _prep(df)
  return (((abs(d["close"] - d["open"]) / (d["high"] - d["low"])) < 0.1) & (
      (d["high"] - np.maximum(d["close"], d["open"])) > (
      3 * abs(d["close"] - d["open"]))) & (
              (np.minimum(d["close"], d["open"]) - d["low"]) > (
              3 * abs(d["close"] - d["open"]))))


# ------------------------------------------------------------------ #
#  MULTI‑BAR CANDLESTICK DETECTORS                                   #
#  All logic mirrors the EODHD Medium article heuristics.            #
# ------------------------------------------------------------------ #
def cs_three_white_soldiers(df):
  d = _prep(df)
  b1 = (d["close"].shift(2) > d["open"].shift(2))
  b2 = (d["close"].shift(1) > d["open"].shift(1))
  b3 = (d["close"] > d["open"])
  # each opens within body of prior and closes higher
  o2_in_body1 = (d["open"].shift(1) < d["close"].shift(2)) & (
      d["open"].shift(1) > d["open"].shift(2))
  o3_in_body2 = (d["open"] < d["close"].shift(1)) & (
      d["open"] > d["open"].shift(1))
  c2_gt_c1 = d["close"].shift(1) > d["close"].shift(2)
  c3_gt_c2 = d["close"] > d["close"].shift(1)
  return b1 & b2 & b3 & o2_in_body1 & o3_in_body2 & c2_gt_c1 & c3_gt_c2


def cs_three_black_crows(df):
  d = _prep(df)
  r1 = (d["close"].shift(2) < d["open"].shift(2))
  r2 = (d["close"].shift(1) < d["open"].shift(1))
  r3 = (d["close"] < d["open"])
  o2_in_body1 = (d["open"].shift(1) > d["close"].shift(2)) & (
      d["open"].shift(1) < d["open"].shift(2))
  o3_in_body2 = (d["open"] > d["close"].shift(1)) & (
      d["open"] < d["open"].shift(1))
  c2_lt_c1 = d["close"].shift(1) < d["close"].shift(2)
  c3_lt_c2 = d["close"] < d["close"].shift(1)
  return r1 & r2 & r3 & o2_in_body1 & o3_in_body2 & c2_lt_c1 & c3_lt_c2


def cs_morning_star(df):
  d = _prep(df)
  day1 = d["close"].shift(2) < d["open"].shift(2)  # long bearish
  day2_gap = d["open"].shift(1) < d["close"].shift(2)
  day2_small = abs(d["close"].shift(1) - d["open"].shift(1)) / (
      d["high"].shift(1) - d["low"].shift(1) + 1e-6) < 0.3
  day3 = d["close"] > d["open"]
  close_into = d["close"] > (d["open"].shift(2) + d["close"].shift(2)) / 2
  return day1 & day2_gap & day2_small & day3 & close_into


def cs_evening_star(df):
  d = _prep(df)
  day1 = d["close"].shift(2) > d["open"].shift(2)  # long bullish
  day2_gap = d["open"].shift(1) > d["close"].shift(2)
  day2_small = abs(d["close"].shift(1) - d["open"].shift(1)) / (
      d["high"].shift(1) - d["low"].shift(1) + 1e-6) < 0.3
  day3 = d["close"] < d["open"]
  close_into = d["close"] < (d["open"].shift(2) + d["close"].shift(2)) / 2
  return day1 & day2_gap & day2_small & day3 & close_into


def cs_bullish_harami(df):
  d = _prep(df)
  prev_bear = d["close"].shift(1) < d["open"].shift(1)
  small_bull = d["close"] > d["open"]
  open_in_prev = (d["open"] > d["close"].shift(1)) & (
      d["open"] < d["open"].shift(1))
  close_in_prev = (d["close"] > d["close"].shift(1)) & (
      d["close"] < d["open"].shift(1))
  return prev_bear & small_bull & open_in_prev & close_in_prev


def cs_bearish_harami(df):
  d = _prep(df)
  prev_bull = d["close"].shift(1) > d["open"].shift(1)
  small_bear = d["close"] < d["open"]
  open_in_prev = (d["open"] < d["close"].shift(1)) & (
      d["open"] > d["open"].shift(1))
  close_in_prev = (d["close"] < d["close"].shift(1)) & (
      d["close"] > d["open"].shift(1))
  return prev_bull & small_bear & open_in_prev & close_in_prev


def cs_piercing_pattern(df):
  d = _prep(df)
  first_bear = d["close"].shift(1) < d["open"].shift(1)
  gap_down = d["open"] < d["low"].shift(1)
  close_above_mid = d["close"] > (d["open"].shift(1) + d["close"].shift(1)) / 2
  close_below_open1 = d["close"] < d["open"].shift(1)
  return first_bear & (
      d["close"] > d["open"]) & gap_down & close_above_mid & close_below_open1


def cs_dark_cloud_cover(df):
  d = _prep(df)
  first_bull = d["close"].shift(1) > d["open"].shift(1)
  gap_up = d["open"] > d["high"].shift(1)
  close_below_mid = d["close"] < (d["open"].shift(1) + d["close"].shift(1)) / 2
  close_above_open1 = d["close"] > d["open"].shift(1)
  return first_bull & (
      d["close"] < d["open"]) & gap_up & close_below_mid & close_above_open1


def detect_candlestick_patterns(df: pd.DataFrame) -> List[Dict]:
  """Wrapper to scan for the single-bar candlestick patterns above."""
  patterns = []
  mapping = {"Hammer": (cs_hammer, "bullish"),
             "Inverted Hammer": (cs_inverted_hammer, "bullish"),
             "Shooting Star": (cs_shooting_star, "bearish"),
             "Hanging Man": (cs_hanging_man, "bearish"),
             "Doji": (cs_doji, "neutral"),
             "Three White Soldiers": (cs_three_white_soldiers, "bullish"),
             "Three Black Crows": (cs_three_black_crows, "bearish"),
             "Morning Star": (cs_morning_star, "bullish"),
             "Evening Star": (cs_evening_star, "bearish"),
             "Bullish Harami": (cs_bullish_harami, "bullish"),
             "Bearish Harami": (cs_bearish_harami, "bearish"),
             "Piercing Pattern": (cs_piercing_pattern, "bullish"),
             "Dark Cloud Cover": (cs_dark_cloud_cover, "bearish"), }
  for name, (func, direction) in mapping.items():
    mask = func(df)
    idxs = np.where(mask)[0]
    for i in idxs:
      date = df.iloc[i]["Date"]
      patterns.append({"start_date": date, "end_date": date, "pattern": name,
                       "direction": direction,
                       "value": 60})  # base score; will be re-scored
  return patterns


def detect_pivots(df: pd.DataFrame, left: int = 3, right: int = 3,
    min_diff: float = 0.005, tolerate_equal: bool = True) -> pd.DataFrame:
  """
  Identify local highs/lows (“pivots”) with configurable tolerance.
  `tolerate_equal=True` allows equality on one side so we catch patterns
  in short/flat datasets.
  """
  # Auto‑shrink the window for tiny data sets
  if len(df) < left + right + 3:
    left = right = max(1, len(df) // 5)

  pivots: List[Dict] = []
  for i in range(left, len(df) - right):
    high = df["High"].iloc[i]
    low = df["Low"].iloc[i]

    if tolerate_equal:
      higher_left = all(
          high >= df["High"].iloc[i - j] for j in range(1, left + 1))
      higher_right = all(
          high >= df["High"].iloc[i + j] for j in range(1, right + 1))
      lower_left = all(low <= df["Low"].iloc[i - j] for j in range(1, left + 1))
      lower_right = all(
          low <= df["Low"].iloc[i + j] for j in range(1, right + 1))
    else:
      higher_left = all(
          high > df["High"].iloc[i - j] for j in range(1, left + 1))
      higher_right = all(
          high > df["High"].iloc[i + j] for j in range(1, right + 1))
      lower_left = all(low < df["Low"].iloc[i - j] for j in range(1, left + 1))
      lower_right = all(
          low < df["Low"].iloc[i + j] for j in range(1, right + 1))

    is_high = (higher_left and higher_right and (high - df["Low"].iloc[i]) /
               df["Low"].iloc[i] > min_diff and (any(
            high > df["High"].iloc[i - j] for j in range(1, left + 1)) or any(
            high > df["High"].iloc[i + j] for j in range(1, right + 1))))
    is_low = (lower_left and lower_right and (
        df["High"].iloc[i] - low) / low > min_diff and (any(
        low < df["Low"].iloc[i - j] for j in range(1, left + 1)) or any(
        low < df["Low"].iloc[i + j] for j in range(1, right + 1))))

    if is_high:
      pivots.append({"Index": i, "Type": "High", "Date": df.iloc[i]["Date"],
                     "Price": high})
    if is_low:
      pivots.append(
          {"Index": i, "Type": "Low", "Date": df.iloc[i]["Date"], "Price": low})

  df_pivots = pd.DataFrame(pivots, columns=["Index", "Type", "Date", "Price"])
  # Ensure required columns exist even when no pivots were found
  if df_pivots.empty:
    df_pivots = pd.DataFrame(columns=["Index", "Type", "Date", "Price"])
  return df_pivots


def detect_engulfing_patterns(df: pd.DataFrame) -> List[Dict]:
  # Early exit if required columns are missing
  required_cols = {"Open", "Close"}
  if not required_cols.issubset(df.columns):
    return []
  patterns = []
  df["Body"] = abs(df["Close"] - df["Open"])
  avg_body = df["Body"].rolling(14).mean()
  for i in range(1, len(df)):
    prev = df.iloc[i - 1]
    curr = df.iloc[i]
    if avg_body.iloc[i] == 0 or curr["Body"] < 0.5 * avg_body.iloc[i]:
      continue
    if prev['Close'] < prev['Open'] and curr['Close'] > curr['Open'] and curr[
      'Open'] < prev['Close'] and curr['Close'] > prev['Open']:
      patterns.append({'start_date': prev['Date'], 'end_date': curr['Date'],
                       'pattern': 'Bullish Engulfing', 'direction': 'bullish',
                       'value': 80})
    elif prev['Close'] > prev['Open'] and curr['Close'] < curr['Open'] and curr[
      'Open'] > prev['Close'] and curr['Close'] < prev['Open']:
      patterns.append({'start_date': prev['Date'], 'end_date': curr['Date'],
                       'pattern': 'Bearish Engulfing', 'direction': 'bearish',
                       'value': 80})
  return patterns


def detect_head_shoulders_pivot(df: pd.DataFrame, pivots: pd.DataFrame,
    volume_col: str = None) -> List[Dict]:
  patterns = []
  highs = pivots[pivots['Type'] == 'High'].reset_index(drop=True)
  lows = pivots[pivots['Type'] == 'Low'].reset_index(drop=True)

  for i in range(2, len(highs)):
    left = highs.iloc[i - 2]
    head = highs.iloc[i - 1]
    right = highs.iloc[i]
    shoulder_diff = abs(left['Price'] - right['Price']) / head['Price']
    if left['Price'] < head['Price'] and right['Price'] < head[
      'Price'] and shoulder_diff < 0.07:
      neckline = min(df['Low'].iloc[head['Index']],
                     df['Low'].iloc[right['Index']])
      segment = df.iloc[left['Index']:right['Index'] + 1]
      breakout = segment['Close'].iloc[-1] < neckline
      volume_support = True
      if volume_col and volume_col in df.columns:
        volume_support = segment[volume_col].iloc[-1] > segment[
          volume_col].mean()
      if breakout and volume_support:
        patterns.append({'start_date': df.iloc[left['Index']]['Date'],
                         'end_date': df.iloc[right['Index']]['Date'],
                         'pattern': 'Head and Shoulders',
                         'direction': 'bearish', 'value': 100})

  for i in range(2, len(lows)):
    left = lows.iloc[i - 2]
    head = lows.iloc[i - 1]
    right = lows.iloc[i]
    shoulder_diff = abs(left['Price'] - right['Price']) / head['Price']
    if left['Price'] > head['Price'] and right['Price'] > head[
      'Price'] and shoulder_diff < 0.07:
      neckline = max(df['High'].iloc[head['Index']],
                     df['High'].iloc[right['Index']])
      segment = df.iloc[left['Index']:right['Index'] + 1]
      breakout = segment['Close'].iloc[-1] > neckline * 0.99
      volume_support = True
      if volume_col and volume_col in df.columns:
        volume_support = segment[volume_col].iloc[-1] > segment[
          volume_col].mean()
      if breakout and volume_support:
        patterns.append({'start_date': df.iloc[left['Index']]['Date'],
                         'end_date': df.iloc[right['Index']]['Date'],
                         'pattern': 'Inverse Head and Shoulders',
                         'direction': 'bullish', 'value': 100})
  return patterns


def detect_double_tops_bottoms_pivot(df: pd.DataFrame, pivots: pd.DataFrame) -> \
    List[Dict]:
  patterns = []
  highs = pivots[pivots['Type'] == 'High'].reset_index(drop=True)
  lows = pivots[pivots['Type'] == 'Low'].reset_index(drop=True)

  min_distance = 3  # Require at least 3 candles between points

  # === Double Top ===
  for i in range(2, len(highs)):
    first = highs.iloc[i - 2]
    mid = highs.iloc[i - 1]
    last = highs.iloc[i]

    if last['Index'] - first['Index'] < min_distance:
      continue

    if np.isclose(first['Price'], last['Price'], rtol=0.03):
      if mid['Price'] < first['Price'] * 0.97:
        neckline1 = df['Low'].iloc[mid['Index']]
        neckline2 = df['Low'].iloc[last['Index']]
        neckline_slope = abs(neckline2 - neckline1) / max(neckline1, neckline2)

        if neckline_slope < 0.02:
          post_segment = df.iloc[last['Index'] + 1: last['Index'] + 4]
          if len(post_segment) >= 2 and post_segment[
            'Close'].min() < neckline1 * 0.99:
            patterns.append(
                {'start_date': first['Date'], 'end_date': last['Date'],
                 'pattern': 'Double Top', 'direction': 'bearish', 'value': 100})

  # === Double Bottom ===
  for i in range(2, len(lows)):
    first = lows.iloc[i - 2]
    mid = lows.iloc[i - 1]
    last = lows.iloc[i]

    if last['Index'] - first['Index'] < min_distance:
      continue

    if np.isclose(first['Price'], last['Price'], rtol=0.03):
      if mid['Price'] > first['Price'] * 1.03:
        neckline1 = df['High'].iloc[mid['Index']]
        neckline2 = df['High'].iloc[last['Index']]
        neckline_slope = abs(neckline2 - neckline1) / max(neckline1, neckline2)

        if neckline_slope < 0.02:
          post_segment = df.iloc[last['Index'] + 1: last['Index'] + 4]
          if len(post_segment) >= 2 and post_segment[
            'Close'].max() > neckline1 * 1.01:
            patterns.append(
                {'start_date': first['Date'], 'end_date': last['Date'],
                 'pattern': 'Double Bottom', 'direction': 'bullish',
                 'value': 100})

  return patterns


def detect_triangle_pivot(df: pd.DataFrame, pivots: pd.DataFrame,
    window: int = 10) -> List[Dict]:
  patterns = []
  highs = pivots[pivots['Type'] == 'High'].reset_index(drop=True)
  lows = pivots[pivots['Type'] == 'Low'].reset_index(drop=True)

  for i in range(len(highs) - window + 1):
    highs_window = highs.iloc[i:i + window]
    lows_window = lows[lows['Index'].between(highs_window['Index'].min(),
                                             highs_window['Index'].max())]
    if len(lows_window) < 2:
      continue

    flat_highs = highs_window['Price'].max() - highs_window[
      'Price'].min() < 0.02 * highs_window['Price'].max()
    rising_lows = all(
        lows_window.iloc[j]['Price'] < lows_window.iloc[j + 1]['Price'] for j in
        range(len(lows_window) - 1))

    if flat_highs and rising_lows:
      start = df.iloc[highs_window['Index'].min()]
      end = df.iloc[highs_window['Index'].max()]
      patterns.append({'start_date': start['Date'], 'end_date': end['Date'],
                       'pattern': 'Ascending Triangle', 'direction': 'bullish',
                       'value': 95})

  return patterns


def detect_wedge_patterns_slope(df: pd.DataFrame, window: int = 10) -> List[
  Dict]:
  patterns: List[Dict] = []
  # If the dataframe lacks standard OHLC columns (e.g. user passed nested‑list
  # JSON incorrectly), just skip wedge detection gracefully.
  required_cols = {"High", "Low", "Open", "Close"}
  if not required_cols.issubset(df.columns):
    return patterns
  min_slope_diff = 0.002
  min_slope_magnitude = 0.001

  for i in range(len(df) - window + 1):
    segment = df.iloc[i:i + window]
    high_slope = slope(segment['High'])
    low_slope = slope(segment['Low'])

    # 🔹 1. Shrinking candle bodies check
    body_sizes = (segment['Close'] - segment['Open']).abs()
    body_trend = slope(body_sizes)
    shrinking_bodies = body_trend < 0

    # 🔹 2. Volume contraction (optional but used if column exists)
    volume_ok = True
    volume_trend = 0
    if 'Volume' in df.columns:
      vol_segment = segment['Volume']
      volume_trend = slope(vol_segment)
      volume_ok = volume_trend < 0

    # 🔹 3. Candle strength filter (avoid fake wedges with strong breakout candles)
    avg_body = body_sizes.mean()
    max_body = body_sizes.max()
    strong_candles = max_body > avg_body * 1.5

    # 🔹 4. Volatility regime check
    volatility = segment['Close'].std()

    # === Rising Wedge ===
    if high_slope > 0 and low_slope > 0 and high_slope < low_slope:
      if (abs(high_slope - low_slope) > min_slope_diff and abs(
          high_slope) > min_slope_magnitude and abs(
          low_slope) > min_slope_magnitude and shrinking_bodies and volume_ok and not strong_candles):
        confidence = 'High' if volume_trend < -0.01 and volatility < 2 else 'Moderate'
        patterns.append({'start_date': segment.iloc[0]['Date'],
                         'end_date': segment.iloc[-1]['Date'],
                         'pattern': 'Rising Wedge', 'direction': 'bearish',
                         'value': 90, 'confidence': confidence})

    # === Falling Wedge ===
    elif high_slope < 0 and low_slope < 0 and high_slope > low_slope:
      if (abs(high_slope - low_slope) > min_slope_diff and abs(
          high_slope) > min_slope_magnitude and abs(
          low_slope) > min_slope_magnitude and shrinking_bodies and volume_ok and not strong_candles):
        confidence = 'High' if volume_trend < -0.01 and volatility < 2 else 'Moderate'
        patterns.append({'start_date': segment.iloc[0]['Date'],
                         'end_date': segment.iloc[-1]['Date'],
                         'pattern': 'Falling Wedge', 'direction': 'bullish',
                         'value': 90, 'confidence': confidence})

  return patterns


def detect_channel_patterns_slope(df: pd.DataFrame, window: int = 10) -> List[
  Dict]:
  patterns: List[Dict] = []
  required_cols = {"High", "Low"}
  if not required_cols.issubset(df.columns):
    return patterns
  for i in range(len(df) - window + 1):
    segment = df.iloc[i:i + window]
    high_slope = slope(segment['High'])
    low_slope = slope(segment['Low'])
    if abs(high_slope - low_slope) < 0.005:
      if high_slope > 0:
        patterns.append({'start_date': segment.iloc[0]['Date'],
                         'end_date': segment.iloc[-1]['Date'],
                         'pattern': 'Channel Up', 'direction': 'bullish',
                         'value': 90})
      elif high_slope < 0:
        patterns.append({'start_date': segment.iloc[0]['Date'],
                         'end_date': segment.iloc[-1]['Date'],
                         'pattern': 'Channel Down', 'direction': 'bearish',
                         'value': 90})
  return patterns


# --------------------------------------------------------------------- #
# Simple, window‑based recognisers – useful when pivot logic is too     #
# restrictive for miniature synthetic test sets.                        #
# --------------------------------------------------------------------- #
def detect_inverse_head_shoulders_raw(df: pd.DataFrame, window: int = 3,
    tolerance: float = 0.05) -> List[Dict]:
  """
  Naïve three‑candles inverse head‑and‑shoulders finder.
  Looks for L‑S‑R where the middle low (head) is deeper than the two
  shoulders and where the shoulder lows are roughly equal.
  """
  patterns: List[Dict] = []
  for i in range(2, len(df)):
    ls, head, rs = df.iloc[i - 2], df.iloc[i - 1], df.iloc[i]
    if abs(ls["Low"] - rs["Low"]) / ((ls["Low"] + rs["Low"]) / 2) <= tolerance:
      if head["Low"] < ls["Low"] and head["Low"] < rs["Low"]:
        patterns.append({"start_date": ls["Date"], "end_date": rs["Date"],
                         "pattern": "Inverse Head and Shoulders",
                         "direction": "bullish", "value": 100, })
  return patterns


def detect_ascending_triangle_raw(df: pd.DataFrame, window: int = 3,
    tolerance: float = 0.02) -> List[Dict]:
  """
  Detect flat‑top / rising‑bottom three‑bar ascending triangles.
  """
  patterns: List[Dict] = []
  for i in range(window - 1, len(df)):
    seg = df.iloc[i - (window - 1): i + 1]
    highs = seg["High"]
    lows = seg["Low"]

    flat_highs = highs.max() - highs.min() <= tolerance * highs.mean()
    rising_lows = all(
        lows.iloc[j] < lows.iloc[j + 1] for j in range(len(lows) - 1))

    if flat_highs and rising_lows:
      patterns.append(
          {"start_date": seg.iloc[0]["Date"], "end_date": seg.iloc[-1]["Date"],
           "pattern": "Ascending Triangle", "direction": "bullish",
           "value": 95, })
  return patterns


def detect_double_bottom_raw(df: pd.DataFrame, window: int = 5,
    tolerance: float = 0.03) -> List[Dict]:
  """
  Sliding‑window double‑bottom detector: low – rally – retest low.
  """
  patterns: List[Dict] = []
  for start in range(len(df) - window + 1):
    seg = df.iloc[start: start + window]
    low1 = seg.iloc[0]
    low2 = seg.iloc[-1]

    # Two lows roughly equal
    if abs(low1["Low"] - low2["Low"]) / low1["Low"] > tolerance:
      continue

    # Mid‑segment high must clearly exceed the lows (neckline breakout)
    if seg["High"][1:-1].max() <= low1["Low"] * (1 + tolerance):
      continue

    patterns.append({"start_date": low1["Date"], "end_date": low2["Date"],
                     "pattern": "Double Bottom", "direction": "bullish",
                     "value": 100, })
  return patterns


def calculate_pattern_score(pattern: dict, df: pd.DataFrame,
    volume_col: str = None) -> float:
  duration = (pd.to_datetime(pattern["end_date"]) - pd.to_datetime(
      pattern["start_date"])).days
  if duration == 0:
    duration = 1

  volume_score = 0
  segment = df.loc[
    (df["Date"] >= pattern["start_date"]) & (df["Date"] <= pattern["end_date"])]
  if volume_col and volume_col in df.columns and not segment.empty:
    avg_volume = segment[volume_col].mean()
    breakout_volume = segment[volume_col].iloc[-1]
    if avg_volume:
      volume_score = (breakout_volume - avg_volume) / avg_volume * 100

  close_start = df.loc[df['Date'] == pattern['start_date'], 'Close']
  close_end = df.loc[df['Date'] == pattern['end_date'], 'Close']
  if close_start.empty or close_end.empty:
    return 0.0  # fallback if data missing

  breakout_strength = abs(close_end.values[0] - close_start.values[0])
  avg_close = df['Close'].mean() if not df['Close'].empty else 1.0

  score = (0.3 * min(duration, 30) / 30 * 100 + 0.3 * max(0, min(volume_score,
                                                                 100)) + 0.4 * min(
      breakout_strength / avg_close * 100, 100))

  # Make the scoring pattern-aware
  pattern_weights = {'Doji': 0.5, 'Hammer': 1.0, 'Shooting Star': 1.0,
                     'Morning Star': 1.2, 'Three White Soldiers': 1.3,
                     'Double Bottom': 1.5, 'Inverse Head and Shoulders': 1.6,
                     'Head and Shoulders': 1.6, 'Double Top': 1.5, }
  weight = pattern_weights.get(pattern["pattern"], 1.0)
  score *= weight

  # --- reliability adjustment ------------------------------------
  reliability = get_pattern_reliability(pattern["pattern"])
  # maps reliability 0.5‑1.0 → multiplier 0.75‑1.0
  score *= (0.5 + 0.5 * reliability)

  return round(score, 2)


def resolve_conflicts(patterns: List[Dict]) -> List[Dict]:
  grouped_by_time = {}
  for pattern in patterns:
    key = (pattern["start_date"], pattern["end_date"])
    if key not in grouped_by_time:
      grouped_by_time[key] = []
    grouped_by_time[key].append(pattern)
  return [max(group, key=lambda p: p["value"]) for group in
          grouped_by_time.values()]


def backtest_pattern_strategy(df: pd.DataFrame, patterns: List[Dict],
    risk: float = 0.01):
  capital = 10000
  position_size = 0
  equity_curve = []

  patterns = ensure_pattern_dates_are_datetime(patterns)
  patterns = [p for p in patterns if
              isinstance(p["start_date"], pd.Timestamp) and isinstance(
                  p["end_date"], pd.Timestamp) and not (
                    pd.isna(p["start_date"]) or pd.isna(p["end_date"]))]
  df['Date'] = pd.to_datetime(df['Date'])
  for i, row in df.iterrows():
    date = row['Date']
    price = row['Close']

    # Check for signal
    signals = [p for p in patterns if p['start_date'] <= date <= p['end_date']]
    for signal in signals:
      if signal['pattern'] == 'Bearish Engulfing' and signal[
        'direction'] == 'bearish':
        entry_price = price
        stop_loss = entry_price * 1.01  # 1% stop
        target = entry_price * 0.98  # 2% profit
        position_size = capital * risk / (stop_loss - entry_price)
        capital -= position_size * entry_price  # enter short

    # Exit logic (simulate 2 bars after signal)
    if i >= 2 and position_size > 0:
      exit_price = price
      pnl = (entry_price - exit_price) * position_size
      capital += position_size * exit_price + pnl
      position_size = 0

    equity_curve.append({'Date': date, 'Equity': capital})

  return pd.DataFrame(equity_curve)


def export_analysis_results(results: Dict[str, any],
    output_dir: str = "output"):
  os.makedirs(output_dir, exist_ok=True)

  # Ensure 'status' column is included in both CSV and JSON
  patterns_df = pd.DataFrame(results["patterns"])
  patterns_df["start_date"] = patterns_df["start_date"].astype(str)
  patterns_df["end_date"] = patterns_df["end_date"].astype(str)
  patterns_df.to_csv(f"{output_dir}/detected_patterns.csv", index=False)
  with open(f"{output_dir}/detected_patterns.json", "w") as f:
    json.dump(patterns_df.to_dict(orient="records"), f, indent=2)

  pd.DataFrame(results["equity_curve"]).to_csv(f"{output_dir}/equity_curve.csv",
                                               index=False)
  with open(f"{output_dir}/equity_curve.json", "w") as f:
    equity_df = results["equity_curve"]
    equity_df["Date"] = equity_df["Date"].astype(str)
    json.dump(equity_df.to_dict(orient="records"), f, indent=2)

  pd.DataFrame(results["next_predictions"]).to_csv(
      f"{output_dir}/next_predictions.csv", index=False)
  with open(f"{output_dir}/next_predictions.json", "w") as f:
    preds_df = pd.DataFrame(results["next_predictions"])
    if not preds_df.empty:
      preds_df["start_date"] = preds_df["start_date"].astype(str)
      preds_df["end_date"] = preds_df["end_date"].astype(str)
    json.dump(preds_df.to_dict(orient="records"), f, indent=2)
  with open(f"{output_dir}/backtest_summary.json", "w") as f:
    json.dump(results["backtest_summary"], f, indent=2)

  # --- cache the full results structure for quick reuse ---------------
  with open(f"{output_dir}/results_cache.json", "w") as f:
    cache = results.copy()
    for k, v in cache.items():
      if isinstance(v, pd.DataFrame):
        cache[k] = v.to_dict(orient="records")

    def convert(obj):
      """
      Recursively make any object JSON‑serialisable.

      * Scalars (str/int/float/bool/None) pass through untouched.
      * NaN / NaT / pandas‐style missing values become None.
      * pd.Timestamp / numpy.datetime64 → ISO‑date strings.
      * Mapping / list / tuple / set containers are walked recursively.
      * Anything still unrecognised is converted to str(..) as a last resort.
      """
      # Fast‑path for already OK primitives
      if obj is None or isinstance(obj, (str, int, float, bool)):
        # Leave np.nan & NaN to the generic pd.isna() handler below
        if obj is None or not (isinstance(obj, float) and pd.isna(obj)):
          return obj
        # --- handle numpy arrays / pandas Series early -----------------
        if isinstance(obj, (np.ndarray, pd.Series)):
          return [convert(v) for v in
                  obj.tolist()]
          # Universal missing-value check (catches pd.NaT, np.nan, etc.)
      try:
        if pd.isna(obj):
          return None
      except (TypeError, ValueError):
        # pd.isna returned an array (e.g. on list/ndarray); let the
        # container-handling branches below process the elements.
        pass

      # Datetime-like → ISO string (date portion only for simplicity)
      if isinstance(obj, (pd.Timestamp, np.datetime64)):
        try:
          return pd.to_datetime(obj).strftime("%Y-%m-%d")
        except Exception:
          return str(obj)

      # Recursively handle common containers
      if isinstance(obj, Mapping):
        return {k: convert(v) for k, v in obj.items()}
      if isinstance(obj, (list, tuple, set)):
        return [convert(v) for v in obj]

      # Fallback: stringify anything else
      return str(obj)

    # Run the deep‑conversion once and serialise with a safe fallback
    json.dump(convert(cache), f, indent=2, default=str)


def plot_analysis_results(results: Dict[str, any]):
  eq_df = pd.DataFrame(results["equity_curve"])
  pred_df = pd.DataFrame(results["next_predictions"])

  plt.figure(figsize=(10, 5))
  plt.plot(eq_df["Date"], eq_df["Equity"], label="Equity Curve", color="blue")
  plt.title("Strategy Backtest Equity Curve")
  plt.xlabel("Date")
  plt.ylabel("Equity")
  plt.grid(True)
  plt.legend()
  plt.xticks(rotation=45)
  plt.tight_layout()
  plt.show()

  if not pred_df.empty:
    print("\n📈 Upcoming Pattern Forecast:")
    for _, row in pred_df.iterrows():
      print(
          f"- {row['start_date']} to {row['end_date']}: {row['expected_pattern']} (Confidence: {row['confidence']})")


def print_summary_report(results: Dict[str, any], show_forecast: bool = True):
  # ------------------------------------------------------------------
  # Filter logic: keep only high‑quality, non‑duplicate patterns
  # ------------------------------------------------------------------
  patterns = results.get("patterns", [])

  def _get_score(pat):
    return pat.get("value", pat.get("score", 0))

  filtered_patterns = [p for p in patterns if
    _get_score(p) >= 1.4  # ► raised from 1.0 → 1.4
    and p.get("status") in ("Confirmed", "Partial")]
  # Show up to the last 12 patterns so single‑bar formations are visible
  for filtered_pattern in filtered_patterns:
    print(
        f"- {filtered_pattern['start_date']} to {filtered_pattern['end_date']}: {filtered_pattern['pattern']} ({filtered_pattern['direction']}, score={_get_score(filtered_pattern)}, status={filtered_pattern.get('status', '?')})")
  if show_forecast:
    if results["next_predictions"]:
      print("\n🔮 Forecast for today:")
      for pred in results["next_predictions"]:
        print(
            f"- {pred['start_date']} to {pred['end_date']}: {pred['expected_pattern']} (Confidence: {pred['confidence']})")
        if "forecast_ohlc" in pred:
          print("  OHLC Forecast:")
          for day in pred["forecast_ohlc"]:
            print(f"    • {day['date']}: "
                  f"O={day['open']} H={day['high']} L={day['low']} C={day['close']}")
    else:
      print("\n🔮 No strong pattern forecast for the next 2 days.")


def analyze_patterns(df: pd.DataFrame, window: int = 5,
    volume_col: str = None) -> Dict[str, any]:
  df = df.copy()

  # --- Auto-flatten synthetic single-cell or single-row frames ----------
  if len(df.columns) == 1 and df.iloc[0].apply(
      lambda x: isinstance(x, list)).all():
    df = pd.DataFrame(df.iloc[0, 0])
  if len(df) == 1 and df.apply(
      lambda col: col.map(lambda x: isinstance(x, dict))).all(axis=None):
    df = pd.DataFrame(df.iloc[0].tolist())

  # -------------------------------------------------
  # 1) Pattern detection
  # -------------------------------------------------
  pivots = detect_pivots(df)

  results: List[Dict] = []
  results += detect_head_shoulders_pivot(df, pivots, volume_col)
  results += detect_double_tops_bottoms_pivot(df, pivots)
  results += detect_triangle_pivot(df, pivots)
  results += detect_wedge_patterns_slope(df, window)
  results += detect_channel_patterns_slope(df, window)
  results += detect_engulfing_patterns(df)
  results += detect_candlestick_patterns(df)
  results += detect_inverse_head_shoulders_raw(df)
  results += detect_ascending_triangle_raw(df)
  results += detect_double_bottom_raw(df)

  # -------------------------------------------------
  # 1.5)  Tag every pattern with its bar-length
  # -------------------------------------------------
  for patt in results:
    try:
      start_idx = df.index[df["Date"] == patt["start_date"]][0]
      end_idx = df.index[df["Date"] == patt["end_date"]][0]
      patt["bars"] = int(end_idx - start_idx + 1)
    except Exception:
      patt["bars"] = 0

  # -------------------------------------------------
  # 2)  Score + resolve conflicts
  # -------------------------------------------------
  for p in results:
    p["value"] = calculate_pattern_score(p, df, volume_col)
  results = resolve_conflicts(results)
  results = sorted(results, key=lambda p: pd.to_datetime(p["start_date"]))
  results = _label_patterns(df, results)  # Confirmed / Partial / Duplicate

  # -------------------------------------------------
  # 2) Back-test (very simple stub)
  # -------------------------------------------------
  equity_curve = backtest_pattern_strategy(df, results)
  backtest_summary = {"start_equity": float(equity_curve.iloc[0]["Equity"]),
                      "end_equity": float(equity_curve.iloc[-1]["Equity"]),
                      "net_pnl": float(
                        equity_curve.iloc[-1]["Equity"] - equity_curve.iloc[0][
                          "Equity"]), }

  # -------------------------------------------------
  # 3) 1-day forecast (sequential bars)
  # -------------------------------------------------
  next_patterns: List[Dict] = []
  if not results or df.empty:
    return next_patterns

  recent_patterns = results[-3:]
  strongest = max(recent_patterns,
                  key=lambda p: p["value"]) if recent_patterns else None
  direction_hint = strongest["direction"] if strongest else "bullish"
  expected_pattern = "Bullish Continuation" if direction_hint == "bullish" else "Downtrend Continuation"

  if strongest:
    score = strongest["value"]
    confidence = "Very High" if score > 80 else "High" if score >= 70 else "Moderate" if score >= 50 else "Low"
  else:
    confidence = "Moderate"

  # True ATR: includes gaps
  df["prev_close"] = df["Close"].shift(1)
  df["tr1"] = df["High"] - df["Low"]
  df["tr2"] = (df["High"] - df["prev_close"]).abs()
  df["tr3"] = (df["Low"] - df["prev_close"]).abs()
  df["TR"] = df[["tr1", "tr2", "tr3"]].max(axis=1)
  atr = df["TR"].tail(5).mean()

  # Momentum calculation (5-bar drift)
  if len(df) >= 6:
    fast_drift = (df["Close"].iloc[-1] - df["Close"].iloc[-6]) / 5
  else:
    fast_drift = 0.0

  momentum_sign = np.sign(fast_drift)
  pattern_sign = 1 if direction_hint == "bullish" else -1
  alignment = momentum_sign * pattern_sign

  # Next weekday
  forecast_dates = []
  curr_date = pd.to_datetime(df["Date"].max())
  while len(forecast_dates) < 1:
    curr_date += timedelta(days=1)
    if curr_date.weekday() < 5:
      forecast_dates.append(curr_date)

  # Short-term EMA
  ema_fast = df["Close"].ewm(span=3).mean().iloc[-1]
  ema_slow = df["Close"].ewm(span=10).mean().iloc[-1]
  ema_diff = ema_fast - ema_slow

  bars = []
  tick = 0.05
  q = lambda v: round(round(v / tick) * tick, 2)
  last_close_price = df["Close"].iloc[-1]

  for d in forecast_dates:
    if alignment < 0 and abs(fast_drift) > 0.05 * atr:
      drive_sign = momentum_sign
    else:
      drive_sign = pattern_sign

    base_mag = max(abs(fast_drift), 0.05 * atr)
    base_mag *= 0.6 if alignment == 0 else 0.8 if alignment < 0 else 1.0
    bias = drive_sign * base_mag

    # Overlay EMA confirmation
    if np.sign(ema_diff) == drive_sign:
      ema_adjustment = np.clip(ema_diff / atr, -0.25, 0.25) * atr
      bias += ema_adjustment

    # Range based on recent candle behavior
    recent = df.tail(10)
    up_offsets = (recent["High"] - recent[["Open", "Close"]].max(axis=1)).abs()
    dn_offsets = (recent[["Open", "Close"]].min(axis=1) - recent["Low"]).abs()
    median_up = up_offsets.median() or 0.3 * atr
    median_dn = dn_offsets.median() or 0.3 * atr

    hi_off = 1.1 * median_up if drive_sign > 0 else 0.9 * median_up
    lo_off = 0.9 * median_dn if drive_sign > 0 else 1.1 * median_dn

    open_price = q(last_close_price)
    close_price = q(open_price + bias)
    high_price = q(max(open_price, close_price) + hi_off)
    low_price = q(min(open_price, close_price) - lo_off)

    if low_price > open_price:
      low_price = q(open_price - tick)
    if high_price < close_price:
      high_price = q(close_price + tick)

    bars.append(
        {"Date": d.strftime("%Y-%m-%d"), "Open": open_price, "High": high_price,
         "Low": low_price, "Close": close_price, })

    last_close_price = close_price

  forecast_window = pd.DataFrame(bars)
  ohlc = [{"date": row["Date"], "open": f"{row['Open']:.2f}",
           "high": f"{row['High']:.2f}", "low": f"{row['Low']:.2f}",
           "close": f"{row['Close']:.2f}"} for _, row in
          forecast_window.iterrows()]

  next_patterns.append({"start_date": forecast_window.iloc[0]["Date"],
                        "end_date": forecast_window.iloc[-1]["Date"],
                        "expected_pattern": expected_pattern,
                        "confidence": confidence, "forecast_ohlc": ohlc, })

  # -------------------------------------------------
  # 4) Return structured results
  # -------------------------------------------------
  return {"patterns": results, "equity_curve": equity_curve,
          "backtest_summary": backtest_summary,
          "next_predictions": next_patterns, }


def generate_evolving_daily_ohlc(intraday_df):
  return {"Open": intraday_df.iloc[0]['Open'],
          "High": intraday_df['High'].max(), "Low": intraday_df['Low'].min(),
          "Close": intraday_df.iloc[-1]['Close']}


# --------------------------------------------------------------------- #
# Post‑detection validation helpers                                      #
# --------------------------------------------------------------------- #
def _breakout_confirmed(df: pd.DataFrame, pattern: Dict) -> bool:
  """
  Simple breakout confirmation one bar AFTER the pattern ends.
  Uses   close > neckline * 1.01  (bullish) or close < neckline * 0.99 (bearish).
  """
  if pattern["pattern"] not in {"Double Bottom", "Double Top",
                                "Inverse Head and Shoulders",
                                "Head and Shoulders", "Ascending Triangle"}:
    return True  # treat other patterns as intrinsically confirmed

  try:
    end_idx = df.index[df["Date"] == pattern["end_date"]][0]
  except IndexError:
    return False

  if end_idx + 1 >= len(df):
    return False  # no bar after the pattern

  next_close = df["Close"].iloc[end_idx + 1]

  # --- Obtain a neckline/flat‑high or low reference ------------------
  start_idx = df.index[df["Date"] == pattern["start_date"]][0]
  segment = df.iloc[start_idx: end_idx + 1]
  if pattern["pattern"] in {"Double Bottom", "Inverse Head and Shoulders",
                            "Ascending Triangle"}:
    neckline = segment["High"].max()
    return next_close > neckline * 1.01
  elif pattern["pattern"] in {"Double Top", "Head and Shoulders"}:
    neckline = segment["Low"].min()
    return next_close < neckline * 0.99
  return True


def _label_patterns(df: pd.DataFrame, patterns: List[Dict]) -> List[Dict]:
  """
  Annotate each pattern with a 'status' field:
    - 'Confirmed' if breakout confirmed
    - 'Duplicate' if time‑overlap with earlier same‑type pattern
    - 'Partial'   otherwise
  """
  labelled: List[Dict] = []
  for p in patterns:
    # Duplicate check
    duplicate = any((p["pattern"] == q["pattern"]) and not (
        pd.to_datetime(p["end_date"]) < pd.to_datetime(
        q["start_date"]) or pd.to_datetime(p["start_date"]) > pd.to_datetime(
        q["end_date"])) for q in labelled)
    if duplicate:
      p["status"] = "Duplicate"
      labelled.append(p)
      continue

    # Breakout confirmation
    p["status"] = "Confirmed" if _breakout_confirmed(df, p) else "Partial"
    labelled.append(p)
  return labelled


def forecast_next_days(historical_data, days=2):
  """
  Generate OHLC forecasts for the next `days` days based on historical OHLC data.

  Parameters:
      historical_data (DataFrame or list-like): Historical OHLC data. It can be a pandas DataFrame
          with columns ['Open','High','Low','Close'], or a list of dicts/tuples in that order.
      days (int): Number of future days to forecast.

  Returns:
      List[dict]: A list of forecasted OHLC values for the next days, where each entry is a dict
      with keys 'Open','High','Low','Close'.
  """
  # Extract OHLC series from the input data
  if hasattr(historical_data, "iloc"):  # pandas DataFrame support
    opens = historical_data["Open"].tolist()
    highs = historical_data["High"].tolist()
    lows = historical_data["Low"].tolist()
    closes = historical_data["Close"].tolist()
  else:
    # Assume list of dict or tuple/list in OHLC order
    opens = [];
    highs = [];
    lows = [];
    closes = []
    for record in historical_data:
      if isinstance(record, dict):
        opens.append(record.get("Open") or record.get("open"))
        highs.append(record.get("High") or record.get("high"))
        lows.append(record.get("Low") or record.get("low"))
        closes.append(record.get("Close") or record.get("close"))
      else:
        # Assume tuple or list [Open, High, Low, Close]
        try:
          o, h, l, c = record
        except Exception:
          raise ValueError("Unsupported data format for historical_data")
        opens.append(o);
        highs.append(h);
        lows.append(l);
        closes.append(c)
  if len(closes) == 0:
    return []

  forecaster = _OHLCForecaster(opens, highs, lows, closes)
  return forecaster.predict(days)


def refine_next_predictions(results: Dict[str, any], df: pd.DataFrame,
    days: int = 1, weight_pattern: float = 0.3,
    weight_volatility: float = 0.7) -> Dict[str, any]:
  """
  Combine the pattern-biased forecast from `analyze_patterns` with a
  pure volatility/ATR forecast (from `forecast_next_days`) to generate
  a more realistic OHLC prediction.

  Parameters
  ----------
  results : dict
      The full results structure returned by `analyze_patterns`.
      Must contain results["next_predictions"].
  df : pd.DataFrame
      The historical OHLC dataframe used for the first pass.
  days : int
      Number of future days to forecast (defaults to 2).
  weight_pattern : float
      Weight assigned to the pattern-biased forecast (0-1).
  weight_volatility : float
      Weight assigned to the pure volatility forecast (0-1).
      `weight_pattern + weight_volatility` should sum to 1.

  Returns
  -------
  Dict[str, any]
      Same `results` object, augmented with
      `results["next_predictions_refined"]`.
  """
  if "next_predictions" not in results or not results["next_predictions"]:
    # Nothing to refine – just copy the field so caller logic is consistent
    results["next_predictions_refined"] = results.get("next_predictions", [])
    return results

  # --- Build volatility-driven baseline forecast --------------------
  baseline = forecast_next_days(df.tail(30), days)  # 30-bar window

  # Convert first-pass pattern forecast to numeric form
  patt_raw = results["next_predictions"][0].get("forecast_ohlc", [])
  if not patt_raw:
    # Fall back to baseline only
    results["next_predictions_refined"] = baseline
    return results

  pattern_cast = []
  for rec in patt_raw:
    # stored as strings; convert to floats
    pattern_cast.append(dict(Open=float(rec["open"]), High=float(rec["high"]),
                             Low=float(rec["low"]), Close=float(rec["close"])))

  # --- Blend the forecasts -----------------------------------------
  refined = []
  prev_close = df["Close"].iloc[-1]  # ensure Open = prev Close constraint
  for i in range(days):
    # Safety in case forecasts differ in length
    p = pattern_cast[min(i, len(pattern_cast) - 1)]
    v = baseline[min(i, len(baseline) - 1)]

    open_price = prev_close
    high_price = (weight_pattern * p["High"] + weight_volatility * v["High"])
    low_price = (weight_pattern * p["Low"] + weight_volatility * v["Low"])
    close_price = (weight_pattern * p["Close"] + weight_volatility * v["Close"])

    # Enforce logical OHLC ordering
    high_price = max(high_price, open_price, close_price)
    low_price = min(low_price, open_price, close_price)

    # Round to 2 decimals for consistency
    refined.append(dict(Open=round(open_price, 2), High=round(high_price, 2),
                        Low=round(low_price, 2), Close=round(close_price, 2)))
    prev_close = refined[-1]["Close"]

  # Attach to the results dictionary
  results["next_predictions_refined"] = refined
  return results


class _OHLCForecaster:
  """Helper class to forecast OHLC values for future days based on recent history."""

  def __init__(self, opens, highs, lows, closes):
    self.opens = opens
    self.highs = highs
    self.lows = lows
    self.closes = closes
    # Pre-compute initial True Ranges and ATR for volatility measures
    self._tr_history = []
    for i in range(1, len(closes)):
      prev_close = closes[i - 1]
      tr = max(highs[i] - lows[i], abs(highs[i] - prev_close),
               abs(lows[i] - prev_close))
      self._tr_history.append(tr)
    # Use last 14 TR values (or fewer if not available) to initialize ATR
    period = 14
    recent_trs = self._tr_history[-period:] if len(
        self._tr_history) >= period else self._tr_history
    self._atr = statistics.mean(recent_trs) if recent_trs else 0.0
    # Store recent close-to-open changes (close - previous close, since open=prev close)
    # which is effectively (close - open) for each day given open = prev close
    self._daily_changes = [c - o for o, c in zip(opens, closes)]
    # Keep last day’s range for shock adjustments
    if highs and lows:
      self._last_range = highs[-1] - lows[-1]
    else:
      self._last_range = 0.0

  def predict(self, days=1):
    """Predict the next `days` OHLC values. Returns a list of dicts."""
    forecasts = []
    for _ in range(days):
      # Previous close is the next open
      prev_close = self.closes[-1]
      open_price = prev_close
      # Determine predicted close change using recent momentum
      predicted_change = 0.0
      if len(self._daily_changes) > 0:
        last_change = self._daily_changes[-1]
        # If the last change was an outlier (e.g. huge drop/spike), use a longer window average
        if self._atr > 0 and abs(last_change) > 1.5 * self._atr:
          window = min(10, len(self._daily_changes))
          predicted_change = statistics.mean(
              self._daily_changes[-window:]) if window > 0 else 0.0
        else:
          window = min(5, len(self._daily_changes))
          predicted_change = statistics.mean(
              self._daily_changes[-window:]) if window > 0 else 0.0
      # Forecasted close
      close_price = open_price + predicted_change

      # Forecasted range (High-Low) using ATR and last range "shock" adjustment
      # Blend current ATR with last observed range to adapt to volatility regime
      if self._atr is None:
        self._atr = 0.0
      # Shock factor: higher value gives more weight to last_range (recent volatility)
      shock_factor = 0.9
      predicted_range = self._atr + (
          self._last_range - self._atr) * shock_factor
      if predicted_range < 0:
        predicted_range = -predicted_range
      # Ensure a minimum range (to avoid zero range predictions)
      if predicted_range < 1e-4:
        predicted_range = 1e-4
      # Ensure the range covers the net change (with a 10% buffer for intraday swings)
      if abs(predicted_change) > predicted_range:
        predicted_range = abs(predicted_change) * 1.1

      # Determine High and Low based on predicted range and day type (up or down)
      high_price = None
      low_price = None
      if close_price >= open_price:
        # Predicted up-day (including flat case)
        net_move = close_price - open_price  # D = upward movement
        R = predicted_range
        r = net_move / R if R != 0 else 0.0  # fraction of range that is net move
        # Estimate pullback fraction: how far off the high the close will be
        if r <= 0.3:
          pullback_frac = 0.5  # small net move -> likely mid-range close (significant pullback)
        elif r >= 0.7:
          # very strong up move -> close near high (minimal pullback)
          pullback_frac = 0.1 * ((1 - r) / 0.3) if r < 1.0 else 0.0
        else:
          # interpolate between 0.5 and 0.1 as r goes from 0.3 to 0.7
          frac_progress = (r - 0.3) / 0.4
          pullback_frac = 0.5 + (0.1 - 0.5) * frac_progress
        pullback_frac = max(0.0, min(0.5, pullback_frac))
        # High and Low computation for up day
        high_price = close_price + pullback_frac * R
        low_price = high_price - R
      else:
        # Predicted down-day
        net_move = open_price - close_price  # D = downward movement
        R = predicted_range
        r = net_move / R if R != 0 else 0.0
        # Estimate bounce fraction: how far off the low the close will be
        if r <= 0.3:
          bounce_frac = 0.5  # small net move -> likely close mid-range (significant bounce off low)
        elif r >= 0.7:
          # very strong down move -> close near low (minimal bounce)
          bounce_frac = 0.1 * ((1 - r) / 0.3) if r < 1.0 else 0.0
        else:
          # interpolate between 0.5 and 0.1 as r goes from 0.3 to 0.7
          frac_progress = (r - 0.3) / 0.4
          bounce_frac = 0.5 + (0.1 - 0.5) * frac_progress
        bounce_frac = max(0.0, min(0.5, bounce_frac))
        # High and Low computation for down day
        low_price = close_price - bounce_frac * R
        high_price = low_price + R

      # Final safety check: ensure logical ordering of Open, High, Low, Close
      high_price = max(high_price, open_price, close_price)
      low_price = min(low_price, open_price, close_price)

      # Round results (if desired, to 2 decimal places for price)
      open_price = round(open_price, 2)
      high_price = round(high_price, 2)
      low_price = round(low_price, 2)
      close_price = round(close_price, 2)

      # Append forecast
      forecasts.append(
          {"Open": open_price, "High": high_price, "Low": low_price,
           "Close": close_price})

      # Update internal state for multi-day forecasting
      # Append the new values to history for subsequent day predictions
      self.opens.append(open_price);
      self.highs.append(high_price)
      self.lows.append(low_price);
      self.closes.append(close_price)
      # Update daily change list (close - open)
      self._daily_changes.append(close_price - open_price)
      # Update last range and ATR (use new predicted day as if it were actual)
      self._last_range = high_price - low_price
      new_tr = self._last_range  # since we treat predicted open == prev close, true range = high-low
      self._tr_history.append(new_tr)
      if len(self._tr_history) > 14:
        # maintain last 14 TR values
        self._tr_history.pop(0)
      # Recalculate ATR as simple moving average of TR_history
      self._atr = statistics.mean(self._tr_history) if self._tr_history else 0.0

    return forecasts

  # ─────────────────────────── Rich-feature day forecast ────────────────────


def build_feature_stack(df_hist: pd.DataFrame,
    df_today_min: pd.DataFrame | None, daily_patterns: list[dict],
    intraday_patterns: list[dict], vwap_trend: str,
    morning_cutoff: str = "10:30") -> dict:
  """
  Assemble a rich dictionary of numeric features harvested from:
      • Daily history (ATR, RSI, pattern bias …)
      • Intraday tape so‑far (realised range, VWAP gap, morn breakout …)
      • Pattern reliability (PATTERN_RELIABILITY weightings)

  Parameters
  ----------
  df_hist : daily OHLC frame (≥ 30 rows recommended)
  df_today_min : intraday 1‑min (or similar) frame; may be None pre‑market.
  daily_patterns / intraday_patterns : output from the detectors
  vwap_trend : 'UP' or 'DOWN' from the VWAP/OBV gauge
  morning_cutoff : time (HH:MM) delimiting the “opening range” window

  Returns
  -------
  dict   – feature → float
  """

  # ---------- DAILY‑FRAME METRICS ---------------------------------
  atr14 = df_hist["High"].sub(df_hist["Low"]).rolling(14).mean().iloc[-1]

  # Simple RSI(14)
  delta = df_hist["Close"].diff()
  gain = delta.clip(lower=0).rolling(14).mean()
  loss = (-delta.clip(upper=0)).rolling(14).mean()
  rs = gain / loss.replace(0, np.nan)
  rsi14 = 100 - 100 / (1 + rs.iloc[-1])

  # Pattern bias and reliability‑adjusted score
  def _bias_score(patts: list[dict]) -> tuple[int, float]:
    bias = 0
    rel_sum = 0.0
    for p in patts:
      sign = 1 if p["direction"] == "bullish" else -1 if p[
                                                           "direction"] == "bearish" else 0
      bias += sign
      rel_sum += sign * get_pattern_reliability(p["pattern"])
    max_n = max(1, len(patts))
    rel_norm = rel_sum / max_n  # −1 … +1
    return bias, rel_norm

  b_intr, rel_intr = _bias_score(intraday_patterns)
  b_daily, rel_daily = _bias_score(daily_patterns)
  rel_score = 0.6 * rel_daily + 0.4 * rel_intr  # weighted blend

  # ---------- INTRADAY (SO‑FAR) METRICS ---------------------------
  high_so_far = low_so_far = vwap_gap_pct = np.nan
  morn_breakout = 0
  morn_range_norm = 0.0

  if df_today_min is not None and not df_today_min.empty:
    # Opening price & VWAP gap
    ohlc_first = df_today_min.iloc[0]
    vwap_now = (df_today_min["Close"] * df_today_min["Volume"]).cumsum().iloc[
                 -1] / df_today_min["Volume"].cumsum().iloc[-1]
    vwap_gap_pct = (ohlc_first["Close"] - vwap_now) / max(1e-6,
                                                          ohlc_first["Close"])

    # Realised extremes
    high_so_far = df_today_min["High"].max()
    low_so_far = df_today_min["Low"].min()

    # ----- Opening‑range stats (to `morning_cutoff`) -------------
    # --- Determine the timestamp marking the end of the opening range ---
    first_day_str = df_today_min["Datetime"].dt.date.iloc[0].strftime(
      "%Y-%m-%d")
    cutoff_ts = pd.to_datetime(f"{first_day_str} {morning_cutoff}")
    open_window = df_today_min[df_today_min["Datetime"] <= cutoff_ts]
    if not open_window.empty:
      or_high = open_window["High"].max()
      or_low = open_window["Low"].min()
      morn_range_norm = (or_high - or_low) / max(1e-6, atr14)
      last_px = df_today_min["Close"].iloc[-1]
      if last_px > or_high * 1.001:
        morn_breakout = 1
      elif last_px < or_low * 0.999:
        morn_breakout = -1

  # ---------- PACK ------------------------------------------------
  return dict(atr14=float(atr14), rsi14=float(rsi14), patt_intr=b_intr,
      patt_daily=b_daily, rel_score=float(rel_score),
      high_so_far=float(high_so_far), low_so_far=float(low_so_far),
      vwap_gap=float(vwap_gap_pct), vwap_trend=1 if vwap_trend == "UP" else -1,
      morn_brk=morn_breakout, morn_rng=morn_range_norm, )


def probabilistic_day_forecast(features: dict, open_price: float,
    base_prob: float = 0.50, alpha: float = 4.0) -> dict:
  """
  Upgraded day-level OHLC forecaster.
  • Adds realised range & drift           (high_so_far / low_so_far, morn_brk)
  • Scales by pattern reliability         (rel_score)
  • Blends ATR with realised intraday vol for the day-range
  """

  atr = max(1e-6, features.get("atr14", 0.0))

  # ---------- 1. helper metrics ----------
  realised_range = max(0.0,
      features.get("high_so_far", 0.0) - features.get("low_so_far", 0.0))
  rng_norm = np.clip(realised_range / atr, 0.0, 2.0)  # 0-2 ATR so far

  drift_dir = features.get("morn_brk", 0)  # -1 / 0 / +1
  rel_score = features.get("rel_score", 0.0)  # -1…+1 (optional)

  # ---------- 2. weighted linear score ----------
  z = 0.0
  z += 0.28 * np.tanh(features.get("patt_daily", 0))
  z += 0.18 * np.tanh(features.get("patt_intr", 0))
  z += 0.12 * features.get("vwap_trend", 0)
  z += 0.12 * ((features.get("rsi14", 50) - 50) / 50)
  z += 0.08 * (-features.get("vwap_gap", 0) * 10)
  z += 0.08 * drift_dir
  z += 0.07 * (1 - rng_norm)  # small a.m. range → room to run
  z += 0.07 * rel_score

  # ---------- 3. probability & confidence ----------
  p_up = 1 / (1 + np.exp(-alpha * z))  # logistic
  direction = "UP" if p_up >= 0.5 else "DOWN"
  confidence = float(np.clip(abs(p_up - 0.5) * 2, 0.0, 0.9))  # 0-0.9

  # ---------- 4. translate into OHLC ----------
  # expected net move (close-open)
  move = (p_up - 0.5) * 0.8 * atr * 2  # net move up to ±0.8 ATR
  close_est = open_price + move

  # day range: half ATR plus a share of what we’ve already printed
  day_range = 0.5 * atr + 0.3 * realised_range
  hi = max(open_price, close_est) + day_range
  lo = min(open_price, close_est) - day_range

  return {"confidence": round(confidence, 2), "O": round(open_price, 2),
    "H": round(hi, 2), "L": round(lo, 2), "C": round(close_est, 2),
    "direction": direction, }
