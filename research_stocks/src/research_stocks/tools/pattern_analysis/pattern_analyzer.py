# pattern_analyzer.py
# -----------------
# Core pattern analysis functions

from typing import List, Dict, Any

import pandas as pd

from .candlestick_patterns import detect_candlestick_patterns
from .chart_patterns import (detect_pivots, detect_head_shoulders_pivot,
                             detect_double_tops_bottoms_pivot,
                             detect_triangle_pivot)
from .utils import get_pattern_reliability, slope


def calculate_pattern_score(pattern: dict, df: pd.DataFrame,
    volume_col: str = None) -> float:
  """
  Calculate a score for a pattern based on various factors.

  Args:
      pattern: Pattern dictionary
      df: DataFrame with OHLC data
      volume_col: Name of volume column if available

  Returns:
      Score value
  """
  # Base reliability
  reliability = get_pattern_reliability(pattern['pattern'])

  single_bar = pattern['pattern'] in [
      'Hammer', 'Inverted Hammer', 'Shooting Star',
      'Hanging Man', 'Doji']

  base_score = reliability * (0.6 if single_bar else 1.0)

  # Pattern height factor (taller patterns are more significant)
  height_factor = pattern.get('height', 0) / df['Close'].mean() * 10

  # Duration factor (longer patterns are more significant)
  start_date = pd.to_datetime(pattern['start_date'])
  end_date = pd.to_datetime(pattern['end_date'])
  duration = (end_date - start_date).days + 1
  duration_factor = min(duration / 10, 1.5)  # Cap at 1.5

  # Volume factor if volume data is available
  volume_factor = 1.0
  if volume_col and volume_col in df.columns:
    # Calculate average volume during pattern vs. before pattern
    pattern_idx = df[(df['Date'] >= pattern['start_date']) & (
          df['Date'] <= pattern['end_date'])].index

    if len(pattern_idx) > 0:
      before_idx = df[df.index < pattern_idx[0]].index[
                   -min(20, len(df[df.index < pattern_idx[0]])):]

      if len(before_idx) > 0:
        avg_vol_pattern = df.loc[pattern_idx, volume_col].mean()
        avg_vol_before = df.loc[before_idx, volume_col].mean()

        if avg_vol_before > 0:
          volume_factor = min(avg_vol_pattern / avg_vol_before,
                              2.0)  # Cap at 2.0

  # Body ratio factor for single candle patterns
  body_factor = 1.0
  if single_bar:
    row = df[df['Date'] == pattern['end_date']]
    if not row.empty:
      body = abs(row['Close'].values[0] - row['Open'].values[0])
      rng = row['High'].values[0] - row['Low'].values[0]
      if rng > 0:
        body_factor = 0.5 + min(body / rng, 1)

  # Trend context factor
  trend_factor = 1.0
  lookback = df[df['Date'] < pattern['start_date']].tail(5)
  if not lookback.empty:
    trend = slope(lookback['Close'])
    if pattern['direction'] == 'bullish' and trend >= 0:
      trend_factor = 0.5
    elif pattern['direction'] == 'bearish' and trend <= 0:
      trend_factor = 0.5

  score = base_score * (1 + height_factor) * duration_factor * \
          volume_factor * body_factor * trend_factor

  # Normalize to a reasonable range (0-5)
  score = min(max(score, 0), 5)

  return score


def resolve_conflicts(patterns: List[Dict]) -> List[Dict]:
  """
  Resolve conflicts between overlapping patterns by keeping the highest-scoring one.

  Args:
      patterns: List of pattern dictionaries

  Returns:
      List of patterns with conflicts resolved
  """
  if not patterns:
    return patterns

  # Sort patterns by score (highest first)
  sorted_patterns = sorted(patterns, key=lambda p: p.get('value', 0),
                           reverse=True)

  # Keep track of which patterns to keep
  keep = [True] * len(sorted_patterns)

  # Check each pattern against higher-scoring patterns
  for i in range(1, len(sorted_patterns)):
    if not keep[i]:
      continue

    p1 = sorted_patterns[i]
    p1_start = pd.to_datetime(p1['start_date'])
    p1_end = pd.to_datetime(p1['end_date'])

    for j in range(i):
      if not keep[j]:
        continue

      p2 = sorted_patterns[j]
      p2_start = pd.to_datetime(p2['start_date'])
      p2_end = pd.to_datetime(p2['end_date'])

      # Check for significant overlap
      overlap_start = max(p1_start, p2_start)
      overlap_end = min(p1_end, p2_end)

      if overlap_start <= overlap_end:
        # Calculate overlap percentage
        p1_days = (p1_end - p1_start).days + 1
        overlap_days = (overlap_end - overlap_start).days + 1

        if overlap_days / p1_days > 0.5:  # More than 50% overlap
          keep[i] = False
          break

  # Return only the patterns to keep
  return [p for i, p in enumerate(sorted_patterns) if keep[i]]


def analyze_patterns(
    symbol: str,
    df_daily: pd.DataFrame,
    df_hourly: pd.DataFrame | None = None,
    *,
    window: int = 5,
    volume_col: str | None = None,
) -> Dict[str, Any]:
  """
  Scan *daily* and (optionally) *hourly* data for candlestick and chart
  patterns and return a unified result object.

  • Each detected pattern carries a ``timeframe`` field: "daily" | "hourly".
  • Daily scan is mandatory; hourly scan runs only when a DataFrame is supplied.
  """
  # ── helper to process one DataFrame ────────────────────────────────
  def _scan_one(df: pd.DataFrame, timeframe: str) -> list[dict]:
    if len(df) < window + 5:
      return []

    local: list[dict] = []
    piv = detect_pivots(df)

    # --- candlesticks -------------------------------------------------
    csticks = detect_candlestick_patterns(df)
    for pname in csticks.columns:
      for i in range(len(csticks)):
        if not csticks.iloc[i][pname]:
          continue

        direction = (
          "bearish"
          if pname in {
            "Shooting Star",
            "Hanging Man",
            "Three Black Crows",
            "Evening Star",
            "Bearish Harami",
          }
          else "bullish"
        )
        pat = {
          "pattern": pname,
          "start_date": df.iloc[max(0, i - 2)]["Date"],
          "end_date": df.iloc[i]["Date"],
          "direction": direction,
          "value": 1.0,
          "status": "Confirmed",
          "timeframe": timeframe,
        }
        if i > 0:
          pat["height"] = abs(
              df.iloc[i]["Close"] - df.iloc[i - 1]["Close"]
          )
        pat["value"] = calculate_pattern_score(pat, df, volume_col)
        local.append(pat)

    # --- geometric patterns -----------------------------------------
    gpats = []
    gpats.extend(detect_head_shoulders_pivot(df, piv))
    gpats.extend(detect_double_tops_bottoms_pivot(df, piv))
    gpats.extend(detect_triangle_pivot(df, piv, window))
    for p in gpats:
      p["timeframe"] = timeframe
      p["value"] = calculate_pattern_score(p, df, volume_col)
    local.extend(gpats)
    return local

  # ── run scans ───────────────────────────────────────────────────────
  patterns: list[dict] = _scan_one(df_daily, "daily")
  if df_hourly is not None and not df_hourly.empty:
    patterns.extend(_scan_one(df_hourly, "hourly"))

  # conflict resolution / sorting
  patterns = resolve_conflicts(patterns)
  patterns.sort(key=lambda p: pd.to_datetime(p["start_date"]))

  # build result object
  results = {
    "symbol": symbol,
    "patterns": patterns,
    "next_prediction": None,
    "stock_data_daily": df_daily.to_dict("records"),
  }
  if df_hourly is not None and not df_hourly.empty:
    results["stock_data_hourly"] = df_hourly.to_dict("records")

  return results