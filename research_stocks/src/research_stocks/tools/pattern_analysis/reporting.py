# reporting.py
# -----------
# Functions for reporting and visualization of pattern analysis results

import json
import os
from datetime import datetime
from typing import Any, Dict

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype


def _convert(obj: Any) -> Any:
  """Convert common pandas/numpy objects for JSON serialization."""
  if isinstance(obj, pd.Timestamp):
    return obj.strftime('%Y-%m-%d')
  elif isinstance(obj, pd.DataFrame):
    df = obj.copy()
    for col in df.columns:
      if is_datetime64_any_dtype(df[col]):
        df[col] = df[col].astype(str)
    return df.to_dict(orient='records')
  elif isinstance(obj, pd.Series):
    ser = obj.copy()
    if is_datetime64_any_dtype(ser):
      ser = ser.astype(str)
    return ser.to_dict()
  elif isinstance(obj, (float, int)) and (pd.isna(obj) or pd.isnull(obj)):
    return None
  elif hasattr(obj, 'tolist'):
    return obj.tolist()
  else:
    return obj


def _convert_recursive(value: Any) -> Any:
  """Recursively convert objects to JSON-serialisable forms."""
  if isinstance(value, dict):
    return {k: _convert_recursive(v) for k, v in value.items()}
  if isinstance(value, list):
    return [_convert_recursive(v) for v in value]
  return _convert(value)


def export_analysis_results(results: Dict[str, Any],
    output_dir: str = "output") -> None:
  """
  Export pattern analysis results to JSON file.

  Args:
      results: Dictionary with analysis results
      output_dir: Directory to save output files
  """
  # Create output directory if it doesn't exist
  os.makedirs(output_dir, exist_ok=True)

  # Convert results to JSON-serializable format
  export_results = _convert_recursive(results)

  # Save to JSON file
  symbol = results.get('symbol', '')
  fileName = 'pattern_analysis_results_' + symbol + '.json'
  output_file = os.path.join(output_dir, fileName)
  with open(output_file, 'w') as f:
    json.dump(export_results, f, indent=2)

  print(f"Analysis results exported to {output_file}")


def print_summary_report(results: Dict[str, Any],
    show_forecast: bool = True) -> None:
  """
  Print a summary report of pattern analysis results.

  Args:
      results: Dictionary with analysis results
      show_forecast: Whether to show forecast information
  """
  if not results or 'patterns' not in results or not results['patterns']:
    print("No patterns detected.")
    return

  # Sort patterns by score
  def _get_score(pat):
    return pat.get('value', 0)

  sorted_patterns = sorted(results['patterns'], key=_get_score, reverse=True)

  # Print pattern summary
  print(f"\nDetected {len(sorted_patterns)} patterns:")
  print("-" * 60)
  print(
    f"{'Pattern':<20} {'Direction':<10} {'Start':<12} {'End':<12} {'Score':<6}")
  print("-" * 60)

  for pattern in sorted_patterns:
    print(f"{pattern['pattern']:<20} {pattern['direction']:<10} "
          f"{pattern['start_date']:<12} {pattern['end_date']:<12} "
          f"{pattern.get('value', 0):.2f}")

  # Print forecast if available and requested
  if show_forecast and 'next_prediction' in results and results[
    'next_prediction']:
    pred = results['next_prediction']
    print("\nForecast for next period:")
    print("-" * 60)

    if 'direction' in pred:
      print(f"Direction: {pred['direction']}")

    if 'confidence' in pred:
      print(f"Confidence: {pred['confidence']:.2f}")

    if all(k in pred for k in ['O', 'H', 'L', 'C']):
      print(f"OHLC: Open={pred['O']:.2f}, High={pred['H']:.2f}, "
            f"Low={pred['L']:.2f}, Close={pred['C']:.2f}")

  print("-" * 60)


def generate_evolving_daily_ohlc(intraday_df: pd.DataFrame) -> Dict[str, float]:
  """
  Generate an evolving daily OHLC row from intraday data.

  Args:
      intraday_df: DataFrame with intraday OHLC data

  Returns:
      Dictionary with OHLC values
  """
  if intraday_df is None or intraday_df.empty:
    return {}

  # Extract date from the first row (assuming 'Date' column has format 'YYYY-MM-DD HH:MM')
  date_str = intraday_df.iloc[0]['Date'].split(' ')[0] if ' ' in \
                                                          intraday_df.iloc[0][
                                                            'Date'] else \
  intraday_df.iloc[0]['Date']

  # Calculate OHLC values
  ohlc = {'Date': date_str, 'Open': intraday_df.iloc[0]['Open'],
    'High': intraday_df['High'].max(), 'Low': intraday_df['Low'].min(),
    'Close': intraday_df.iloc[-1]['Close']}

  # Add volume if available
  if 'Volume' in intraday_df.columns:
    ohlc['Volume'] = intraday_df['Volume'].sum()

  return ohlc


def export_enhanced_results(results: Dict[str, Any],
    output_dir: str = "output/model_enhanced") -> None:
  """Export enhanced multi-timeframe results to JSON."""
  os.makedirs(output_dir, exist_ok=True)

  today = datetime.now().strftime("%d-%m-%Y")
  symbol = results.get("symbol", "")

  date_dir = os.path.join(output_dir, today)
  os.makedirs(date_dir, exist_ok=True)

  filename = os.path.join(date_dir,
                          f"{symbol}_Json_{today.split('-')[0]}{today.split('-')[1]}")

  export_data = _convert_recursive(results)

  with open(filename, 'w') as f:
    json.dump(export_data, f, indent=2)

  print(f"📝 Enhanced results saved to {filename}")
