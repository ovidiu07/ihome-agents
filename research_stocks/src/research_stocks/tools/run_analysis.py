#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# run_analysis.py
# Entry-point script for the pattern-analysis tool-chain
# ---------------------------------------------------------------------------

from __future__ import annotations

import os
import sys
from pathlib import Path
from .pattern_analysis.today_forecast import make_today_forecast
import pandas as pd

# Optional dotenv support ----------------------------------------------------
try:
  from dotenv import load_dotenv
except ImportError:  # pragma: no cover
  def load_dotenv() -> None:  # type: ignore
    print("⚠️  python-dotenv not installed ‑ set env variables manually.")

# ───────── Internal imports (adjust package path if necessary) ───────────────
from .pattern_analysis.data_fetchers import fetch_intraday_bars, \
  fetch_daily_history, fetch_hourly_history
from .pattern_analysis.pattern_filters import (drop_duplicates,
                                              suppress_nearby_hits,
                                              cluster_and_keep_best,
                                              filter_patterns_by_criteria,
                                              remove_duplicates_by_status, )
from .pattern_analysis.forecast_utils import (get_intraday_bias, get_daily_bias,
                                             blended_forecast,
                                             calculate_vwap_obv_trend,
                                             calculate_atr, )
from .pattern_analysis.pattern_analyzer import analyze_patterns
from .pattern_analysis.reporting import (export_analysis_results,
                                        print_summary_report,
                                        generate_evolving_daily_ohlc, )
from .pattern_analysis.forecasting import (refine_next_predictions,
                                          probabilistic_day_forecast,  # ← new forecaster signature
)
from .pattern_analysis.intraday_factors import collect_intraday_factors


# ---------------------------------------------------------------------------


def main(symbol: str = "NVDA") -> None:
  """
  Run the complete pattern analysis pipeline for a given stock symbol.

  This function orchestrates the entire pattern analysis workflow:
  1. Fetches historical data (daily, hourly, and intraday)
  2. Analyzes patterns in the data at different time scales
  3. Filters and processes the identified patterns
  4. Generates forecasts using multiple methods (ensemble and probabilistic)
  5. Exports the results to JSON files

  The function requires a Polygon API key to be set in the environment variables
  to fetch the necessary market data.

  Args:
      symbol: Stock ticker symbol to analyze (default: "NVDA")
              Will be automatically converted to uppercase

  Returns:
      None: Results are exported to files and printed to console

  Side Effects:
      - Creates/updates JSON files in the output directory
      - Prints analysis summaries and forecasts to the console
      - May create intraday factor snapshots
  """
  # ─── Environment / configuration ───────────────────────────────────────
  load_dotenv()
  poly_key: str | None = os.getenv("POLYGON_KEY")
  if not poly_key:
    print("❌  POLYGON_KEY not set in environment variables.", file=sys.stderr)
    return

  symbol = symbol.upper()
  daily_lookback = "1mo"  # daily history to pull
  hourly_lookback = "5d"  # daily history to pull
  mc_paths = 2_000  # Monte-Carlo paths for probabilistic forecast

  # ─── Fetch historical data ────────────────────────────────────────────
  df_daily_history = fetch_daily_history(symbol, period=daily_lookback)

  df_hourly_history = fetch_hourly_history(symbol, period=hourly_lookback)

  # Intraday (today)
  df_today_min = fetch_intraday_bars(symbol, poly_key, limit=150)

  # Check if we have intraday data available for analysis
  if df_today_min is None or df_today_min.empty:
    print("⚠️  Intraday pattern scan skipped — no data.")
    # Use only historical data if no intraday data is available
    df_daily = df_daily_history.tail(180)  # Last 180 days of daily data
    df_hourly = df_hourly_history.tail(180)  # Last 180 hours of hourly data
    intraday_filtered: list[dict] = []  # Empty list as no patterns can be found
  else:
    print("\n🔍 Scanning intraday patterns …")
    # Generate a synthetic daily candle from intraday data
    df_today = pd.DataFrame([generate_evolving_daily_ohlc(df_today_min)])

    # Append the synthetic today's candle to historical data
    df_daily = pd.concat([df_daily_history, df_today], ignore_index=True)
    df_hourly = pd.concat([df_hourly_history, df_today], ignore_index=True)

    # Analyze intraday patterns with a 7-bar lookback window
    raw_intraday = \
    analyze_patterns(symbol, df_today_min, window=7)["patterns"]

    # Filter and clean up the intraday patterns:
    # 1. Apply criteria: min_value >= 1.2, status="Confirmed", min_duration_minutes >= 20
    # 2. Suppress nearby hits with a 10-minute gap to avoid duplicates
    intraday_filtered = suppress_nearby_hits(
        filter_patterns_by_criteria(raw_intraday, min_value=1.2,
            status="Confirmed", min_duration_minutes=20, ), gap=10, )

    # Report intraday patterns if any were found
    if intraday_filtered:
      print("\n🧠 Intraday pattern summary:")
      print_summary_report({"patterns": intraday_filtered}, show_forecast=False)
    else:
      print("ℹ️  No qualifying intraday patterns found.")

  # ─── Daily-candle pattern analysis ─────────────────────────────────────
  # Analyze patterns in daily data with a 5-bar lookback window
  # This also uses hourly data for additional context
  results = analyze_patterns(symbol, df_daily, df_hourly, window=5)

  # Process the daily patterns through a multi-step filtering pipeline:
  # 1. Drop exact duplicates from the raw pattern list
  # 2. Remove patterns marked as "Duplicate" status
  # 3. Cluster similar patterns and keep only the best one from each cluster
  #    with an overlap threshold of 0.7 (70% similarity)
  daily_patterns = cluster_and_keep_best(
      remove_duplicates_by_status(drop_duplicates(results["patterns"]),
          status_to_remove="Duplicate",  # ← fixed keyword
      ), overlap=0.7, )
  results["patterns"] = daily_patterns

  # Enhance the patterns with next-day prediction information
  # This adds 'name' and 'direction' keys to each pattern
  results = refine_next_predictions(results, df_daily)

  # Save the initial results to JSON file
  export_analysis_results(results)

  # Print a summary of the daily patterns if any were found
  if results["patterns"]:
    print("\n📊 Daily pattern summary:")
    print_summary_report(results, show_forecast=False)
  else:
    print("ℹ️  No patterns in daily data.")

  # ─── Ancillary trend / ensemble bias ───────────────────────────────────
  # Calculate VWAP (Volume-Weighted Average Price) and OBV (On-Balance Volume) trend
  # Use intraday data if available, otherwise fall back to daily history
  vwap_trend = calculate_vwap_obv_trend(
      df_today_min if df_today_min is not None and not df_today_min.empty else df_daily_history)

  # Calculate Average True Range (ATR) with a 14-period lookback
  # ATR measures market volatility and is used to adjust forecast confidence
  atr14 = calculate_atr(df_daily_history, period=14)

  # Generate a blended forecast that combines multiple signals:
  # 1. Intraday pattern direction bias
  # 2. Daily pattern direction bias
  # 3. VWAP/OBV trend
  # 4. Volatility context from ATR
  ensemble = blended_forecast(
      intraday_direction=get_intraday_bias(intraday_filtered),
      daily_direction=get_daily_bias(daily_patterns), 
      vwap_trend=vwap_trend,
      atr=atr14, )
  print(f"\n🔮 Ensemble forecast: {ensemble}")

  # ─── NEW: Probabilistic next-day forecast ──────────────────────────────
  # Generate a Monte Carlo simulation-based forecast using:
  # 1. Historical OHLC data
  # 2. Active patterns detected in the analysis
  # 3. A specified number of Monte Carlo paths (2,000)
  # 4. Beta_k parameter (1.0) to scale the drift component
  day_fcast = probabilistic_day_forecast(ohlc_df=df_daily,
      active_patterns=results["patterns"], num_mc_paths=mc_paths,
      # feel free to tune
      beta_k=1.0,  # drift scaling parameter for the stochastic process
  )

  # Extract the OHLC prediction from the forecast results
  ohlc = day_fcast["ohlc"]

  # Print the probabilistic forecast details:
  # - Overall bias direction
  # - Probability of upward movement
  # - Confidence level
  # - Predicted OHLC values
  # - 80% confidence interval for price movement
  print(f"\n🔮 Probabilistic forecast → bias: {day_fcast['bias']}, "
        f"P(up)={day_fcast['prob_up']:.2f}, conf={day_fcast['confidence']:.0%}\n"
        f"    O={ohlc['o']:.2f}  H={ohlc['h']:.2f}  "
        f"L={ohlc['l']:.2f}  C={ohlc['c']:.2f}"
        f"  (80 % interval: {day_fcast['interval_80']})")

  # Prepare data for today's forecast by combining:
  # - Daily stock data
  # - Hourly stock data
  # - Detected patterns
  today_blob = {
    "stock_data_daily":   df_daily.to_dict("records"),  # Convert DataFrame to dict records
    "stock_data_hourly":  df_hourly.to_dict("records"),
    "patterns":           results["patterns"],
  }

  # Generate an intraday forecast for today based on the combined data
  results["today_forecast"] = make_today_forecast(today_blob)

  # ─── Collect intraday factor snapshot ─────────────────────────────────-
  # Ensure the next_prediction dictionary exists in the results
  if "next_prediction" not in results:
    results["next_prediction"] = {}

  # Synchronize the Open, Low, and Close values with the Monte Carlo output
  results["next_prediction"]["O"] = day_fcast["ohlc"]["o"]
  results["next_prediction"]["L"] = day_fcast["ohlc"]["l"]
  results["next_prediction"]["C"] = day_fcast["ohlc"]["c"]

  # Save the updated results with all forecasts to JSON
  export_analysis_results(results)

  # Collect additional intraday factors and save them to the output directory
  try:
    factors_path = collect_intraday_factors(symbol, Path("output"))
    print(f"\n📝 Intraday factors saved to {factors_path}")
  except Exception as exc:
    print(f"⚠️  Failed to collect intraday factors: {exc}")


# ---------------------------------------------------------------------------
# Script entry point - allows direct execution from command line
if __name__ == "__main__":
  import sys
  # Get the stock symbol from command line argument or use NVDA as default
  input_symbol = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
  # Run the main analysis pipeline with the provided symbol
  main(input_symbol)
