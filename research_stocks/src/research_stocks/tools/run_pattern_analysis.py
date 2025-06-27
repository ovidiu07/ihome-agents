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
                                      generate_evolving_daily_ohlc,
                                      build_feature_stack,
                                      probabilistic_day_forecast, )


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


# ───────────────────── Extra filtering / bias helpers ──────────────────────
def suppress_nearby_hits(patterns: list[dict], gap: int = 10) -> list[dict]:
  """
  Remove any pattern whose start is fewer than `gap` bars after the previous
  occurrence *of the same pattern type*.
  Patterns MUST be sorted by start_date before calling this helper.
  """
  if not patterns:
    return patterns
  patterns = sorted(patterns, key=lambda p: p["start_date"])
  kept: list[dict] = [patterns[0]]
  for p in patterns[1:]:
    prev = kept[-1]
    same = p["pattern"] == prev["pattern"]
    delta = (pd.to_datetime(p["start_date"]) - pd.to_datetime(
        prev["end_date"])).seconds / 60
    if not (same and delta < gap):
      kept.append(p)
  return kept


def cluster_and_keep_best(patterns: list[dict], overlap: float = 0.7) -> list[
  dict]:
  """
  On the daily list, collapse patterns that overlap > `overlap` fraction in
  time; keep only the highest‑score (“value”) item of each cluster.
  """
  if not patterns:
    return patterns
  patterns = sorted(patterns, key=lambda p: p["start_date"])
  clusters: list[list[dict]] = []
  for p in patterns:
    placed = False
    for c in clusters:
      last = c[0]  # clusters are homogeneous enough
      # --- compute overlap in days (always non‑negative) ---
      s1, e1 = pd.to_datetime(p["start_date"]), pd.to_datetime(p["end_date"])
      s2, e2 = pd.to_datetime(last["start_date"]), pd.to_datetime(
          last["end_date"])

      # raw overlap length (may be negative if no intersection)
      overlap_len = (min(e1, e2) - max(s1, s2)).days + 1

      # clamp to ≥ 0 so we can safely compare with integers
      inter = max(0, overlap_len)

      # length (in days) of the shorter pattern – also ≥ 1
      shorter = min((e1 - s1).days + 1, (e2 - s2).days + 1)
      if shorter and inter / shorter >= overlap:
        c.append(p)
        placed = True
        break
    if not placed:
      clusters.append([p])

  # pick best‑scoring representative
  best = [max(c, key=lambda x: x["value"]) for c in clusters]
  return best


def get_intraday_bias(patterns: list[dict]) -> int:
  """Return +1 for net bullish, ‑1 for net bearish, 0 otherwise."""
  if not patterns:
    return 0
  bulls = sum(p["direction"] == "bullish" for p in patterns)
  bears = sum(p["direction"] == "bearish" for p in patterns)
  return (bulls > bears) - (bears > bulls)


def get_daily_bias(patterns: list[dict]) -> int:
  """Same as above but for daily frame."""
  return get_intraday_bias(patterns)


def blended_forecast(intraday_direction: int, daily_direction: int,
    vwap_trend: str, atr: float) -> str:
  """
  Very simple ensemble: majority vote of three signals.
  Returns 'UP', 'DOWN' or 'NEUTRAL'.
  """
  votes = [intraday_direction, daily_direction, 1 if vwap_trend == "UP" else -1]
  score = sum(votes)
  if score > 1:
    return f"UP (target +{atr:.2f})"
  if score < -1:
    return f"DOWN (target -{atr:.2f})"
  return "NEUTRAL"


# ────────────────────────────────── Main ───────────────────────────────────
def main() -> None:
  global df_today
  load_dotenv()
  poly_key = os.getenv("POLYGON_KEY")
  if not poly_key:
    print("❌  POLYGON_KEY not set in environment variables.")
    return

  # ── Daily history (12 months minus current session) ──
  nvda = yf.Ticker("SPY")
  df_hist = nvda.history(period="12mo", interval="1d").iloc[:-1].copy()
  df_hist.reset_index(inplace=True)
  df_hist["Date"] = df_hist["Date"].dt.strftime("%Y-%m-%d")

  # ── Intraday bars (today) ──
  df_today_min = fetch_intraday_bars("SPY", poly_key, limit=150)

  if df_today_min is None or df_today_min.empty:
    print("⚠️  Skipping intraday pattern scan — no data.")
    df_combined = df_hist.tail(180)
    intraday_filtered = []
  else:
    print("\n🔍 Running pattern analysis on earliest data for today…")

    # ── 1‑minute pattern scan (20‑bar window, strict filters) ──
    raw_intraday = analyze_patterns(df_today_min, window=20)["patterns"]

    intraday_filtered = [p for p in raw_intraday if
                         p.get("value", 0) >= 1.2 and p.get(
                             "status") == "Confirmed" and (
                             pd.to_datetime(p["end_date"]) - pd.to_datetime(
                             p["start_date"])).seconds / 60 >= 20]
    intraday_filtered = suppress_nearby_hits(intraday_filtered, gap=10)

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

  # 3️⃣ collapse heavily‑overlapping daily patterns
  daily_patterns = cluster_and_keep_best(daily_patterns, overlap=0.7)
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
    df_vol = df_today_min if (
        df_today_min is not None and not df_today_min.empty) else df_hist

  if "Volume" in df_vol.columns and not df_vol["Volume"].isnull().all():
    # VWAP — stays a Series
    vwap = (df_vol["Close"] * df_vol["Volume"]).cumsum() / df_vol[
      "Volume"].cumsum()

    # OBV — force a Series (np.where returns ndarray)
    obv_raw = np.where(df_vol["Close"].diff().fillna(0) >= 0, df_vol["Volume"],
                       -df_vol["Volume"]).cumsum()
    obv = pd.Series(obv_raw, index=df_vol.index)

    trend = ("UP" if (
        df_vol["Close"].iloc[-1] > vwap.iloc[-1] and obv.iloc[-1] > obv.iloc[
      0]) else "DOWN")

    # Section which takes data known so far and builds another forecast based on it ; Refined forecast
    features = build_feature_stack(df_hist, df_today_min, daily_patterns,
                                   intraday_filtered, trend)
    today_open = df_today_min.iloc[0][
      "Open"] if df_today_min is not None and not df_today_min.empty else \
      df_hist.iloc[-1]["Open"]

  # ── Ensemble forecast ──
  atr14 = df_hist["High"].sub(df_hist["Low"]).rolling(14).mean().iloc[-1]
  ensemble = blended_forecast(
      intraday_direction=get_intraday_bias(intraday_filtered),
      daily_direction=get_daily_bias(daily_patterns), vwap_trend=trend,
      atr=atr14)
  print(f"\n🔮 Ensemble forecast: {ensemble}")
  print_summary_report(results, show_forecast=True)
  print("\n🔮 OHLC for today’s earliest intraday bars:")
  print(df_today.head(1).to_string(index=False))
  print(f"\n📈 VWAP/OBV trend hint: {trend}")
  day_fcast = probabilistic_day_forecast(features, today_open)
  print(f"\n🔮 Prob-weighted forecast ({day_fcast['direction']}, "
        f"conf {day_fcast['confidence']:.0%}) "
        f"O={day_fcast['O']} H={day_fcast['H']} L={day_fcast['L']} C={day_fcast['C']}")


# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
  main()
