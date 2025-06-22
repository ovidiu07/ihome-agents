import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import linregress
from typing import List, Dict


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
          "pattern": "Inverse Head and Shoulders", "direction": "bullish",
          "value": 100, })
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
      "pattern": "Double Bottom", "direction": "bullish", "value": 100, })
  return patterns


def calculate_pattern_score(pattern: dict, df: pd.DataFrame,
    volume_col: str = None) -> float:
  duration = (pd.to_datetime(pattern["end_date"]) - pd.to_datetime(
      pattern["start_date"])).days
  volume_score = 0
  if volume_col and volume_col in df.columns:
    segment = df.loc[(df["Date"] >= pattern["start_date"]) & (
        df["Date"] <= pattern["end_date"])]
    avg_volume = segment[volume_col].mean()
    breakout_volume = segment[volume_col].iloc[-1]
    volume_score = (
                       breakout_volume - avg_volume) / avg_volume * 100 if avg_volume else 0
  breakout_strength = abs(
      df[df['Date'] == pattern['end_date']]['Close'].values[0] -
      df[df['Date'] == pattern['start_date']]['Close'].values[0])
  score = (0.3 * min(duration, 30) / 30 * 100 + 0.3 * max(0, min(volume_score,
                                                                 100)) + 0.4 * min(
      breakout_strength / df['Close'].mean() * 100, 100))
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
  patterns_df.to_csv(f"{output_dir}/detected_patterns.csv", index=False)
  with open(f"{output_dir}/detected_patterns.json", "w") as f:
    json.dump(patterns_df.to_dict(orient="records"), f, indent=2)

  pd.DataFrame(results["equity_curve"]).to_csv(f"{output_dir}/equity_curve.csv",
                                               index=False)
  with open(f"{output_dir}/equity_curve.json", "w") as f:
    json.dump(results["equity_curve"].to_dict(orient="records"), f, indent=2)

  pd.DataFrame(results["next_predictions"]).to_csv(
      f"{output_dir}/next_predictions.csv", index=False)
  with open(f"{output_dir}/next_predictions.json", "w") as f:
    preds_df = pd.DataFrame(results["next_predictions"])
    if not preds_df.empty:
      preds_df["start_date"] = preds_df["start_date"].astype(str)
      preds_df["end_date"] = preds_df["end_date"].astype(
        str)  # Save back-test summary
  with open(f"{output_dir}/backtest_summary.json", "w") as f:
    json.dump(results["backtest_summary"], f, indent=2)
    json.dump(preds_df.to_dict(orient="records"), f, indent=2)


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


def print_summary_report(results: Dict[str, any]):
  print("\n🧠 Pattern Recognition Summary:")
  for pattern in results["patterns"][-5:]:
    print(
        f"- {pattern['start_date']} to {pattern['end_date']}: {pattern['pattern']} ({pattern['direction']}, score={pattern['value']}, status={pattern.get('status', '?')})")

  print("\n💰 Backtest Summary:")
  b = results["backtest_summary"]
  print(
    f"Start Equity: ${b['start_equity']:.2f} → End Equity: ${b['end_equity']:.2f} "
    f"(Net PnL: ${b['net_pnl']:.2f})")

  if results["next_predictions"]:
    print("\n🔮 Forecast for Next 2 Days:")
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
  results += detect_inverse_head_shoulders_raw(df)
  results += detect_ascending_triangle_raw(df)
  results += detect_double_bottom_raw(df)

  # Score + resolve conflicts
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
    "end_equity": float(equity_curve.iloc[-1]["Equity"]), "net_pnl": float(
      equity_curve.iloc[-1]["Equity"] - equity_curve.iloc[0]["Equity"]), }

  # -------------------------------------------------
  # 3) 2-day forecast (sequential bars)
  # -------------------------------------------------
  next_patterns: List[Dict] = []
  if results:
    recent_patterns = results[-3:]
    strongest = max(recent_patterns,
                    key=lambda p: p["value"]) if recent_patterns else None
    direction_hint = strongest["direction"] if strongest else "bullish"
    expected_pattern = (
      "Bullish Continuation" if direction_hint == "bullish" else "Downtrend Continuation")
    confidence = "High" if strongest and strongest[
      "value"] >= 70 else "Moderate"

    # ATR over the last ≤5 bars
    atr_period = min(5, len(df))
    atr = (df["High"].tail(atr_period) - df["Low"].tail(atr_period)).mean()

    # Next two weekday dates
    forecast_dates: List[pd.Timestamp] = []
    curr_date = pd.to_datetime(df["Date"].max())
    while len(forecast_dates) < 2:
      curr_date += timedelta(days=1)
      if curr_date.weekday() < 5:  # Monday-Friday only
        forecast_dates.append(curr_date)

    # Sequential bar construction
    bars = []
    last_close_price = df["Close"].iloc[-1]  # previous real close
    for d in forecast_dates:
      bias = atr * 0.25 if direction_hint == "bullish" else -atr * 0.25
      open_price = round(last_close_price, 2)
      close_price = round(open_price + bias, 2)
      high_price = round(max(open_price, close_price) + atr * 0.25, 2)
      low_price = round(min(open_price, close_price) - atr * 0.25, 2)
      bars.append({"Date": d.strftime("%Y-%m-%d"), "Open": open_price,
        "High": high_price, "Low": low_price, "Close": close_price, })
      last_close_price = close_price  # next bar opens at prior close

    forecast_window = pd.DataFrame(bars)

    # String-format OHLC for nice printing
    ohlc = [{"date": row["Date"], "open": f"{row['Open']:.2f}",
      "high": f"{row['High']:.2f}", "low": f"{row['Low']:.2f}",
      "close": f"{row['Close']:.2f}", } for _, row in
      forecast_window.iterrows()]

    next_patterns.append({"start_date": forecast_window.iloc[0]["Date"],
      "end_date": forecast_window.iloc[-1]["Date"],
      "expected_pattern": expected_pattern, "confidence": confidence,
      "forecast_ohlc": ohlc, })

  # -------------------------------------------------
  # 4) Return structured results
  # -------------------------------------------------
  return {"patterns": results, "equity_curve": equity_curve,
    "backtest_summary": backtest_summary, "next_predictions": next_patterns, }


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
