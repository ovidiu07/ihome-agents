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
  """Run the complete pattern analysis pipeline."""
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

  if df_today_min is None or df_today_min.empty:
    print("⚠️  Intraday pattern scan skipped — no data.")
    df_daily = df_daily_history.tail(180)
    df_hourly = df_hourly_history.tail(180)
    intraday_filtered: list[dict] = []
  else:
    print("\n🔍 Scanning intraday patterns …")
    df_today = pd.DataFrame([generate_evolving_daily_ohlc(df_today_min)])
    df_daily = pd.concat([df_daily_history, df_today], ignore_index=True)
    df_hourly = pd.concat([df_hourly_history, df_today], ignore_index=True)
    raw_intraday = \
    analyze_patterns(symbol, df_today_min, window=7)["patterns"]

    intraday_filtered = suppress_nearby_hits(
        filter_patterns_by_criteria(raw_intraday, min_value=1.2,
            status="Confirmed", min_duration_minutes=20, ), gap=10, )

    if intraday_filtered:
      print("\n🧠 Intraday pattern summary:")
      print_summary_report({"patterns": intraday_filtered}, show_forecast=False)
    else:
      print("ℹ️  No qualifying intraday patterns found.")

  # ─── Daily-candle pattern analysis ─────────────────────────────────────
  results = analyze_patterns(symbol, df_daily, df_hourly, window=5)

  daily_patterns = cluster_and_keep_best(
      remove_duplicates_by_status(drop_duplicates(results["patterns"]),
          status_to_remove="Duplicate",  # ← fixed keyword
      ), overlap=0.7, )
  results["patterns"] = daily_patterns

  # Refine to next-day predictions (provides 'name'/'direction' keys)
  results = refine_next_predictions(results, df_daily)
  export_analysis_results(results)

  # Daily pattern report
  if results["patterns"]:
    print("\n📊 Daily pattern summary:")
    print_summary_report(results, show_forecast=False)
  else:
    print("ℹ️  No patterns in daily data.")

  # ─── Ancillary trend / ensemble bias ───────────────────────────────────
  vwap_trend = calculate_vwap_obv_trend(
      df_today_min if df_today_min is not None and not df_today_min.empty else df_daily_history)
  atr14 = calculate_atr(df_daily_history, period=14)

  ensemble = blended_forecast(
      intraday_direction=get_intraday_bias(intraday_filtered),
      daily_direction=get_daily_bias(daily_patterns), vwap_trend=vwap_trend,
      atr=atr14, )
  print(f"\n🔮 Ensemble forecast: {ensemble}")

  # ─── NEW: Probabilistic next-day forecast ──────────────────────────────
  day_fcast = probabilistic_day_forecast(ohlc_df=df_daily,
      active_patterns=results["patterns"], num_mc_paths=mc_paths,
      # feel free to tune
      beta_k=1.0,  # drift scaling
  )

  ohlc = day_fcast["ohlc"]
  print(f"\n🔮 Probabilistic forecast → bias: {day_fcast['bias']}, "
        f"P(up)={day_fcast['prob_up']:.2f}, conf={day_fcast['confidence']:.0%}\n"
        f"    O={ohlc['o']:.2f}  H={ohlc['h']:.2f}  "
        f"L={ohlc['l']:.2f}  C={ohlc['c']:.2f}"
        f"  (80 % interval: {day_fcast['interval_80']})")

  # ─── Collect intraday factor snapshot ─────────────────────────────────-
  # --- sync just O and L with Monte-Carlo output ------------------------------
  if "next_prediction" not in results:
    results["next_prediction"] = {}

  results["next_prediction"]["O"] = day_fcast["ohlc"]["o"]
  results["next_prediction"]["L"] = day_fcast["ohlc"]["l"]
  results["next_prediction"]["C"] = day_fcast["ohlc"]["c"]

  export_analysis_results(results)        # re-write JSON
  try:
    factors_path = collect_intraday_factors(symbol, Path("output"))
    print(f"\n📝 Intraday factors saved to {factors_path}")
  except Exception as exc:
    print(f"⚠️  Failed to collect intraday factors: {exc}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
  import sys
  input_symbol = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
  main(input_symbol)
