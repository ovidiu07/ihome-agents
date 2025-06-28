#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# run_analysis.py
# Entry-point script for the pattern-analysis tool-chain
# ---------------------------------------------------------------------------

from __future__ import annotations

import os
import sys

import pandas as pd

# Optional dotenv support ----------------------------------------------------
try:
  from dotenv import load_dotenv
except ImportError:  # pragma: no cover
  def load_dotenv() -> None:  # type: ignore
    print("⚠️  python-dotenv not installed ‑ set env variables manually.")

# ───────── Internal imports (adjust package path if necessary) ───────────────
from pattern_analysis.data_fetchers import fetch_intraday_bars, \
  fetch_daily_history
from pattern_analysis.pattern_filters import (drop_duplicates,
                                              suppress_nearby_hits,
                                              cluster_and_keep_best,
                                              filter_patterns_by_criteria,
                                              remove_duplicates_by_status, )
from pattern_analysis.forecast_utils import (get_intraday_bias, get_daily_bias,
                                             blended_forecast,
                                             calculate_vwap_obv_trend,
                                             calculate_atr, )
from pattern_analysis.pattern_analyzer import analyze_patterns
from pattern_analysis.reporting import (export_analysis_results,
                                        print_summary_report,
                                        generate_evolving_daily_ohlc, )
from pattern_analysis.forecasting import (refine_next_predictions,
                                          probabilistic_day_forecast,  # ← new forecaster signature
)


# ---------------------------------------------------------------------------


def main() -> None:
  """Run the complete pattern analysis pipeline."""
  # ─── Environment / configuration ───────────────────────────────────────
  load_dotenv()
  poly_key: str | None = os.getenv("POLYGON_KEY")
  if not poly_key:
    print("❌  POLYGON_KEY not set in environment variables.", file=sys.stderr)
    return

  symbol = "MSFT"  # ← adapt to taste
  lookback = "12mo"  # daily history to pull
  mc_paths = 2_000  # Monte-Carlo paths for probabilistic forecast

  # ─── Fetch historical data ────────────────────────────────────────────
  df_hist = fetch_daily_history(symbol, period=lookback)

  # Intraday (today)
  df_today_min = fetch_intraday_bars(symbol, poly_key, limit=150)

  if df_today_min is None or df_today_min.empty:
    print("⚠️  Intraday pattern scan skipped — no data.")
    df_combined = df_hist.tail(180)
    df_summary = df_combined.tail(30)
    intraday_filtered: list[dict] = []
  else:
    print("\n🔍 Scanning intraday patterns …")
    df_today = pd.DataFrame([generate_evolving_daily_ohlc(df_today_min)])
    df_combined = pd.concat([df_hist.tail(180), df_today], ignore_index=True)
    df_summary = df_combined.tail(30)
    raw_intraday = \
    analyze_patterns(symbol, df_today_min, df_summary, window=20)["patterns"]

    intraday_filtered = suppress_nearby_hits(
        filter_patterns_by_criteria(raw_intraday, min_value=1.2,
            status="Confirmed", min_duration_minutes=20, ), gap=10, )

    if intraday_filtered:
      print("\n🧠 Intraday pattern summary:")
      print_summary_report({"patterns": intraday_filtered}, show_forecast=False)
    else:
      print("ℹ️  No qualifying intraday patterns found.")

  # ─── Daily-candle pattern analysis ─────────────────────────────────────
  results = analyze_patterns(symbol, df_combined, df_summary, window=5)

  daily_patterns = cluster_and_keep_best(
      remove_duplicates_by_status(drop_duplicates(results["patterns"]),
          status_to_remove="Duplicate",  # ← fixed keyword
      ), overlap=0.7, )
  results["patterns"] = daily_patterns

  # Refine to next-day predictions (provides 'name'/'direction' keys)
  results = refine_next_predictions(results, df_combined)
  export_analysis_results(results)

  # Daily pattern report
  if results["patterns"]:
    print("\n📊 Daily pattern summary:")
    print_summary_report(results, show_forecast=False)
  else:
    print("ℹ️  No patterns in daily data.")

  # ─── Ancillary trend / ensemble bias ───────────────────────────────────
  vwap_trend = calculate_vwap_obv_trend(
      df_today_min if df_today_min is not None and not df_today_min.empty else df_hist)
  atr14 = calculate_atr(df_hist, period=14)

  ensemble = blended_forecast(
      intraday_direction=get_intraday_bias(intraday_filtered),
      daily_direction=get_daily_bias(daily_patterns), vwap_trend=vwap_trend,
      atr=atr14, )
  print(f"\n🔮 Ensemble forecast: {ensemble}")

  # ─── NEW: Probabilistic next-day forecast ──────────────────────────────
  day_fcast = probabilistic_day_forecast(ohlc_df=df_combined,
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


# ---------------------------------------------------------------------------
if __name__ == "__main__":
  main()
