#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# run_analysis.py
# Entry-point script for the pattern-analysis tool-chain
# ---------------------------------------------------------------------------

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

# Optional dotenv support ----------------------------------------------------
try:
  from dotenv import load_dotenv
except ImportError:  # pragma: no cover
  def load_dotenv() -> None:  # type: ignore
    print("⚠️  python-dotenv not installed ‑ set env variables manually.")

# ───────── Internal imports (adjust package path if necessary) ───────────────
from .pattern_analysis.data_fetchers import (
  fetch_intraday_bars,
  fetch_daily_history,
  fetch_hourly_data,
  fetch_minutes_data,
  fetch_polygon_intraday,
)
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
from .pattern_analysis.reporting import (
  export_analysis_results,
  print_summary_report,
  generate_evolving_daily_ohlc,
  export_enhanced_results,
)
from .pattern_analysis.forecasting import (
  refine_next_predictions,
  probabilistic_day_forecast,
  probabilistic_timeframe_forecast,
)
from .pattern_analysis.intraday_factors import collect_intraday_factors


# ---------------------------------------------------------------------------


def main(symbol: str = "NVDA") -> None:
  """Run the complete pattern analysis pipeline with multi-timeframe support."""
  load_dotenv()
  poly_key: str | None = os.getenv("POLYGON_KEY")
  if not poly_key:
    print("❌  POLYGON_KEY not set in environment variables.", file=sys.stderr)
    return

  symbol = symbol.upper()
  lookback_daily = "3mo"
  lookback_hourly = 20
  lookback_minutes = 10
  mc_paths = 2_000

  df_daily_hist = fetch_daily_history(symbol, period=lookback_daily)
  df_hourly_hist = fetch_hourly_data(symbol, days=lookback_hourly)

  df_minutes = fetch_polygon_intraday(symbol, poly_key, interval=15, days=lookback_minutes)
  if df_minutes.empty:
    print("⚠️ Falling back to yfinance for 15-minute data")
    df_minutes = fetch_minutes_data(symbol, interval=15, days=lookback_minutes)

  df_today_min = fetch_intraday_bars(symbol, poly_key, limit=150)

  # ── Daily analysis ───────────────────────────────────────────────────
  df_combined = df_daily_hist.tail(180)
  df_summary = df_combined.tail(30)
  results = analyze_patterns(symbol, df_combined, df_summary, window=5)
  daily_patterns = cluster_and_keep_best(
      remove_duplicates_by_status(drop_duplicates(results["patterns"]),
          status_to_remove="Duplicate"), overlap=0.7)
  results["patterns"] = daily_patterns
  results = refine_next_predictions(results, df_combined)
  export_analysis_results(results)

  # ── Hourly analysis ──────────────────────────────────────────────────
  hourly_results = analyze_patterns(symbol, df_hourly_hist, df_hourly_hist.tail(48), window=12)
  hourly_patterns = cluster_and_keep_best(
      remove_duplicates_by_status(drop_duplicates(hourly_results["patterns"]),
          status_to_remove="Duplicate"), overlap=0.7)
  hourly_results["patterns"] = hourly_patterns
  hourly_results = refine_next_predictions(hourly_results, df_hourly_hist,
                                           weight_pattern=0.5, weight_volatility=0.5)

  # ── 15-minute analysis ───────────────────────────────────────────────
  if not df_minutes.empty:
    minutes_results = analyze_patterns(symbol, df_minutes, df_minutes.tail(48), window=16)
    minutes_patterns = cluster_and_keep_best(
        remove_duplicates_by_status(drop_duplicates(minutes_results["patterns"]),
            status_to_remove="Duplicate"), overlap=0.7)
    minutes_results["patterns"] = minutes_patterns
    minutes_results = refine_next_predictions(minutes_results, df_minutes,
                                              weight_pattern=0.6, weight_volatility=0.4)
  else:
    minutes_results = {"patterns": [], "next_prediction": None, "symbol": symbol}

  # ── Probabilistic forecasts ─────────────────────────────────────────--
  day_fcast = probabilistic_day_forecast(
      ohlc_df=df_combined,
      active_patterns=results["patterns"],
      num_mc_paths=mc_paths,
      beta_k=1.0,
  )

  hour_fcast = probabilistic_timeframe_forecast(
      ohlc_df=df_hourly_hist,
      active_patterns=hourly_results["patterns"],
      num_mc_paths=mc_paths,
      beta_k=0.8,
  )

  if not df_minutes.empty:
    minute_fcast = probabilistic_timeframe_forecast(
        ohlc_df=df_minutes,
        active_patterns=minutes_results["patterns"],
        num_mc_paths=mc_paths,
        beta_k=0.6,
    )
  else:
    minute_fcast = None

  enhanced_results = {
      "symbol": symbol,
      "patterns": results["patterns"],
      "hourly_patterns": hourly_results["patterns"],
      "minutes_patterns": minutes_results["patterns"],
      "next_prediction": results["next_prediction"],
      "hourly_prediction": hourly_results["next_prediction"],
      "minutes_prediction": minutes_results["next_prediction"],
      "stock_data": df_summary.to_dict('records'),
      "hourly_data": df_hourly_hist.tail(48).to_dict('records'),
      "minutes_data": df_minutes.tail(48).to_dict('records') if not df_minutes.empty else [],
      "probabilistic_forecasts": {
          "daily": day_fcast,
          "hourly": hour_fcast,
          "minutes": minute_fcast if minute_fcast else {"message": "No minute data available"}
      },
      "news_headlines": results.get("news_headlines", [])
  }

  export_enhanced_results(enhanced_results)

  print("\n📊 Daily pattern summary:")
  print_summary_report(results, show_forecast=True)

  print("\n⏱️ Hourly pattern summary:")
  print_summary_report(hourly_results, show_forecast=True)

  if not df_minutes.empty:
    print("\n⏲️ 15-minute pattern summary:")
    print_summary_report(minutes_results, show_forecast=True)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
  import sys
  input_symbol = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
  main(input_symbol)
