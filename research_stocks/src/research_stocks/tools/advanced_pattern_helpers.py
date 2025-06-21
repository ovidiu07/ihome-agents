import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.stats import linregress
from typing import List, Dict


def slope(series: pd.Series) -> float:
  x = np.arange(len(series))
  y = series.values
  return linregress(x, y).slope


def detect_pivots(df: pd.DataFrame, left: int = 3, right: int = 3,
    min_diff: float = 0.005) -> pd.DataFrame:
  pivots = []
  for i in range(left, len(df) - right):
    high = df['High'].iloc[i]
    low = df['Low'].iloc[i]
    is_high = all(
        high > df['High'].iloc[i - j] for j in range(1, left + 1)) and all(
        high > df['High'].iloc[i + j] for j in range(1, right + 1)) and (
                    high - df['Low'].iloc[i]) / df['Low'].iloc[i] > min_diff
    is_low = all(
        low < df['Low'].iloc[i - j] for j in range(1, left + 1)) and all(
        low < df['Low'].iloc[i + j] for j in range(1, right + 1)) and (
                   df['High'].iloc[i] - low) / low > min_diff
    if is_high:
      pivots.append({'Index': i, 'Type': 'High', 'Date': df.iloc[i]['Date'],
                     'Price': high})
    if is_low:
      pivots.append(
          {'Index': i, 'Type': 'Low', 'Date': df.iloc[i]['Date'], 'Price': low})
  return pd.DataFrame(pivots)


def detect_engulfing_patterns(df: pd.DataFrame) -> List[Dict]:
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
        'pattern': 'Bullish Engulfing', 'direction': 'bullish', 'value': 80})
    elif prev['Close'] > prev['Open'] and curr['Close'] < curr['Open'] and curr[
      'Open'] > prev['Close'] and curr['Close'] < prev['Open']:
      patterns.append({'start_date': prev['Date'], 'end_date': curr['Date'],
        'pattern': 'Bearish Engulfing', 'direction': 'bearish', 'value': 80})
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
          'pattern': 'Head and Shoulders', 'direction': 'bearish',
          'value': 100})

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
          'pattern': 'Inverse Head and Shoulders', 'direction': 'bullish',
          'value': 100})
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
                  'pattern': 'Double Top', 'direction': 'bearish',
                  'value': 100})

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
        'pattern': 'Ascending Triangle', 'direction': 'bullish', 'value': 95})

  return patterns


def detect_wedge_patterns_slope(df: pd.DataFrame, window: int = 10) -> List[
  Dict]:
  patterns = []
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
          'end_date': segment.iloc[-1]['Date'], 'pattern': 'Rising Wedge',
          'direction': 'bearish', 'value': 90, 'confidence': confidence})

    # === Falling Wedge ===
    elif high_slope < 0 and low_slope < 0 and high_slope > low_slope:
      if (abs(high_slope - low_slope) > min_slope_diff and abs(
          high_slope) > min_slope_magnitude and abs(
          low_slope) > min_slope_magnitude and shrinking_bodies and volume_ok and not strong_candles):
        confidence = 'High' if volume_trend < -0.01 and volatility < 2 else 'Moderate'
        patterns.append({'start_date': segment.iloc[0]['Date'],
          'end_date': segment.iloc[-1]['Date'], 'pattern': 'Falling Wedge',
          'direction': 'bullish', 'value': 90, 'confidence': confidence})

  return patterns


def detect_channel_patterns_slope(df: pd.DataFrame, window: int = 10) -> List[
  Dict]:
  patterns = []
  for i in range(len(df) - window + 1):
    segment = df.iloc[i:i + window]
    high_slope = slope(segment['High'])
    low_slope = slope(segment['Low'])
    if abs(high_slope - low_slope) < 0.005:
      if high_slope > 0:
        patterns.append({'start_date': segment.iloc[0]['Date'],
          'end_date': segment.iloc[-1]['Date'], 'pattern': 'Channel Up',
          'direction': 'bullish', 'value': 90})
      elif high_slope < 0:
        patterns.append({'start_date': segment.iloc[0]['Date'],
          'end_date': segment.iloc[-1]['Date'], 'pattern': 'Channel Down',
          'direction': 'bearish', 'value': 90})
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

  pd.DataFrame(results["patterns"]).to_csv(
    f"{output_dir}/detected_patterns.csv", index=False)
  with open(f"{output_dir}/detected_patterns.json", "w") as f:
    json.dump(results["patterns"], f, indent=2)

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
      preds_df["end_date"] = preds_df["end_date"].astype(str)
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
      f"- {pattern['start_date']} to {pattern['end_date']}: {pattern['pattern']} ({pattern['direction']}, score={pattern['value']})")

  print("\n💰 Backtest Summary:")
  eq_df = pd.DataFrame(results["equity_curve"])
  start_eq = eq_df.iloc[0]["Equity"]
  end_eq = eq_df.iloc[-1]["Equity"]
  print(
    f"Start Equity: ${start_eq:.2f} → End Equity: ${end_eq:.2f} (Net PnL: ${end_eq - start_eq:.2f})")

  if results["next_predictions"]:
    print("\n🔮 Forecast for Next 2 Days:")
    for pred in results["next_predictions"]:
      print(
        f"- {pred['start_date']} to {pred['end_date']}: {pred['expected_pattern']} (Confidence: {pred['confidence']})")
      if "forecast_ohlc" in pred:
        print("  OHLC Forecast:")
        for day in pred["forecast_ohlc"]:
          print(
            f"    • {day['date']}: O={day['open']} H={day['high']} L={day['low']} C={day['close']}")
  else:
    print("\n🔮 No strong pattern forecast for the next 2 days.")


def analyze_patterns(df: pd.DataFrame, window: int = 5, volume_col: str = None) -> Dict[str, any]:
  df = df.copy()

  # Detect pivots
  pivots = detect_pivots(df)

  # Collect all pattern detections
  results = []
  results += detect_head_shoulders_pivot(df, pivots, volume_col)
  results += detect_double_tops_bottoms_pivot(df, pivots)
  results += detect_triangle_pivot(df, pivots)
  results += detect_wedge_patterns_slope(df, window)
  results += detect_channel_patterns_slope(df, window)
  results += detect_engulfing_patterns(df)

  # Score each pattern
  for pattern in results:
    pattern["value"] = calculate_pattern_score(pattern, df, volume_col)

  # Resolve overlapping/conflicting patterns
  results = resolve_conflicts(results)

  # Sort by start_date chronologically
  results = sorted(results, key=lambda p: pd.to_datetime(p["start_date"]))

  # Run backtest simulation
  equity_curve = backtest_pattern_strategy(df, results)

  # Forecast likely patterns for the next 2 days
  next_patterns = []
  if results:
    recent_patterns = results[-3:]  # Use last 3 for forecasting context
    last_pattern_end = pd.to_datetime(results[-1]["end_date"])

    # Filter bars after last pattern end
    future_df = df[pd.to_datetime(df["Date"]) > last_pattern_end]
    forecast_window = future_df.head(2)

    for pattern in recent_patterns:
      if pattern['pattern'] in ["Rising Wedge", "Bearish Engulfing"] and pattern['direction'] == 'bearish':
        expected_pattern = 'Downtrend Continuation'
      elif pattern['pattern'] in ["Falling Wedge", "Double Bottom", "Inverse Head and Shoulders"] and pattern['direction'] == 'bullish':
        expected_pattern = 'Bullish Continuation'
      else:
        continue

      # Build OHLC block
      ohlc = []
      for _, row in forecast_window.iterrows():
        ohlc.append({
          'date': row['Date'],
          'open': row['Open'],
          'high': row['High'],
          'low': row['Low'],
          'close': row['Close']
        })

      forecast = {
        'start_date': forecast_window.iloc[0]['Date'] if not forecast_window.empty else df.iloc[-1]['Date'],
        'end_date': forecast_window.iloc[-1]['Date'] if len(forecast_window) > 1 else df.iloc[-1]['Date'],
        'expected_pattern': expected_pattern,
        'confidence': 'High' if pattern['value'] >= 70 else 'Moderate',
        'forecast_ohlc': ohlc
      }

      next_patterns.append(forecast)

  return {
    "patterns": results,
    "equity_curve": equity_curve,
    "next_predictions": next_patterns
  }