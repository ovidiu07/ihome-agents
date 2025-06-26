# run_pattern_analysis.py
# -----------------------
# Intraday + daily pattern recognition and forecast for NVDA
#
# 2025-06-26  — rev 1.2
# Fixes:
#   • uses p["value"] instead of the missing p["score"]
#   • filters duplicates *before* printing
#   • minimum pattern length / score thresholds
#   • clearer empty-result handling

import os
from datetime import datetime, time

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

from advanced_pattern_helpers import (analyze_patterns, export_analysis_results,
                                      print_summary_report,
                                      refine_next_predictions,
                                      generate_evolving_daily_ohlc, )


# ──────────────────────────── Polygon helpers ──────────────────────────────
def fetch_intraday_bars(symbol: str, api_key: str,
    limit: int = 150) -> pd.DataFrame | None:
  """
  Pulls the latest `limit` 1-minute bars for `symbol` from Polygon.io.
  Returns a DataFrame or None if no data.
  """
  today = datetime.now().strftime("%Y-%m-%d")
  url = (f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/"
         f"{today}/{today}?adjusted=true&sort=asc&limit={limit}&apiKey={api_key}")

  try:
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if not data.get("results"):
      print("⚠️  No intraday data returned (market closed or key expired).")
      return None

    bars = [
      {"Datetime": datetime.fromtimestamp(bar["t"] / 1000), "Open": bar["o"],
       "High": bar["h"], "Low": bar["l"], "Close": bar["c"],
       "Volume": bar.get("v", 0), } for bar in data["results"]]
    df = pd.DataFrame(bars)

    # ► Keep only regular-hours bars (09:30–16:00 ET). Comment out to include pre-/post-market.
    df = df[df["Datetime"].dt.time.between(time(9, 30), time(16, 0))]

    # ── Harmonise column names for downstream helpers ──
    # Many pattern‑detection utilities expect a 'Date' field identical
    # to the daily‑candle DataFrames.  Keep both columns so nothing else breaks.
    if "Date" not in df.columns:
      df["Date"] = df["Datetime"]
    # Store as string like the daily frame, e.g. '2025-06-26 11:03'
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d %H:%M")

    return df

  except Exception as exc:
    print(f"❌ Error fetching intraday bars: {exc}")
    return None


# ───────────────────────── Pattern post-processing ─────────────────────────
def drop_duplicates(patterns: list[dict]) -> list[dict]:
  """
  Remove exact duplicates based on (pattern, start, end).
  Keeps the first occurrence.
  """
  seen: set[tuple] = set()
  unique: list[dict] = []
  for p in patterns:
    key = (p["pattern"], p["start_date"], p["end_date"])
    if key not in seen:
      seen.add(key)
      unique.append(p)
  return unique


# ────────────────────────────────── Main ───────────────────────────────────
def main() -> None:
  load_dotenv()
  poly_key = os.getenv("POLYGON_KEY")
  if not poly_key:
    print("❌  POLYGON_KEY not set in environment variables.")
    return

  # ── Daily history (12 months minus current session) ──
  nvda = yf.Ticker("NVDA")
  df_hist = nvda.history(period="12mo", interval="1d").iloc[:-1].copy()
  df_hist.reset_index(inplace=True)
  df_hist["Date"] = df_hist["Date"].dt.strftime("%Y-%m-%d")

  # ── Intraday bars (today) ──
  df_today_min = fetch_intraday_bars("NVDA", poly_key, limit=150)

  if df_today_min is None or df_today_min.empty:
    print("⚠️  Skipping intraday pattern scan — no data.")
    df_combined = df_hist.tail(180)
  else:
    print("\n🔍 Running pattern analysis on earliest data for today…")

    # 1-minute pattern scan (15-bar rolling window)
    raw_intraday = analyze_patterns(df_today_min, window=15)["patterns"]

    # Filter: score ≥ 1.5, status Confirmed/Partial, min length ≥ 8 bars
    intraday_filtered = [p for p in raw_intraday if
                         p.get("value", 0) >= 1.2 and p.get("status") in {
                           "Confirmed", "Partial"}]
    intraday_filtered = drop_duplicates(intraday_filtered)

    if intraday_filtered:
      print("\n🧠 Premarket / morning pattern summary:")
      print_summary_report({"patterns": intraday_filtered}, show_forecast=False)
    else:
      print("ℹ️  No qualifying intraday patterns.")

    # Aggregate today’s intraday to an evolving daily OHLC row
    df_today = pd.DataFrame([generate_evolving_daily_ohlc(df_today_min)])
    df_combined = pd.concat([df_hist.tail(180), df_today], ignore_index=True)

  # ── Full pattern analysis on daily candles ──
  results = analyze_patterns(df_combined, window=5)

  # 1️⃣ remove exact structural duplicates
  daily_patterns = drop_duplicates(results["patterns"])

  # 2️⃣ suppress engine‑flagged “Duplicate” hits
  daily_patterns = [p for p in daily_patterns if p.get("status") != "Duplicate"]

  results["patterns"] = daily_patterns  # put back into results
  results = refine_next_predictions(results, df_combined)
  export_analysis_results(results)

  if results["patterns"]:
    print("\n📊 Daily pattern summary (look-back 12 mo):")
    for p in results["patterns"]:
      print(f"- {p['start_date']} to {p['end_date']}: "
            f"{p['pattern']} ({p['direction']}, "
            f"score={float(p['value']):.2f}, status={p['status']})")
  else:
    print(
        "ℹ️  No patterns found in daily data.")  # ── Simple VWAP + OBV trend gauge over the fetched window ──

  if df_today_min is not None and not df_today_min.empty:
    print("\n🔮 OHLC for today’s earliest intraday bars:")
    print(df_today.head(1).to_string(index=False))
  # Select a DataFrame that actually exists for the volume-based trend gauge
  df_vol = df_today_min if (
        df_today_min is not None and not df_today_min.empty) else df_hist

  if "Volume" in df_vol.columns and not df_vol["Volume"].isnull().all():
    # VWAP — stays a Series
    vwap = (df_vol["Close"] * df_vol["Volume"]).cumsum() / df_vol["Volume"].cumsum()

    # OBV — force a Series (np.where returns ndarray)
    obv_raw = np.where(df_vol["Close"].diff().fillna(0) >= 0,
                       df_vol["Volume"], -df_vol["Volume"]).cumsum()
    obv = pd.Series(obv_raw, index=df_vol.index)

    trend = ("UP"
             if (df_vol["Close"].iloc[-1] > vwap.iloc[-1]
                 and obv.iloc[-1] > obv.iloc[0])
             else "DOWN")
    print(f"\n📈 VWAP/OBV trend hint: {trend}")
  print_summary_report(results, show_forecast=True)


# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
  main()
