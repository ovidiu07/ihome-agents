# reporting.py
# -----------
# Functions for reporting and visualization of pattern analysis results

import json
import os
from typing import Dict, Any

import pandas as pd


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
  def convert(obj):
    if isinstance(obj, pd.Timestamp):
      return obj.strftime('%Y-%m-%d')
    elif isinstance(obj, pd.DataFrame):
      return obj.to_dict(orient='records')
    elif isinstance(obj, pd.Series):
      return obj.to_dict()
    elif isinstance(obj, (float, int)) and (pd.isna(obj) or pd.isnull(obj)):
      return None
    elif hasattr(obj, 'tolist'):  # For numpy arrays
      return obj.tolist()
    else:
      return obj

  # Create a copy of results to avoid modifying the original
  export_results = {}

  # Convert each item in results
  for key, value in results.items():
    if isinstance(value, list):
      export_results[key] = [{k: convert(v) for k, v in item.items()} for item
        in value] if value else []
    elif isinstance(value, dict):
      export_results[key] = {k: convert(v) for k, v in value.items()}
    else:
      export_results[key] = convert(value)

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
