import os
from datetime import datetime

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

from advanced_pattern_helpers import (analyze_patterns, export_analysis_results,
                                      print_summary_report,
                                      refine_next_predictions,
                                      generate_evolving_daily_ohlc)


def main():
  load_dotenv()
  key = os.getenv("POLYGON_KEY")
  if not key:
    print("❌ POLYGON_KEY not set in environment variables.")
    return

  # Get historical daily candles (excluding current day)
  nvda = yf.Ticker("NVDA")
  df_hist = nvda.history(period="12mo", interval="1d")
  df_hist.reset_index(inplace=True)
  df_hist['Date'] = df_hist['Date'].dt.strftime('%Y-%m-%d')
  df_hist = df_hist[:-1]  # Exclude current day

  # Fetch last 100 1-min bars from Polygon
  df_intraday_today = fetch_intraday_bars("NVDA", key, limit=100)

  if df_intraday_today is None or df_intraday_today.empty:
    print("⚠️ Skipping intraday enhancement — no data.")
    df_combined = df_hist[-180:]
  else:
    print("\n🔍 Running pattern analysis on premarket data (before 09:30 ET)...")
    premarket_results = analyze_patterns(df_intraday_today, window=3)
    print_summary_report(premarket_results, show_forecast=False)
    # Convert today's intraday bars to evolving daily OHLC
    df_today = pd.DataFrame([generate_evolving_daily_ohlc(df_intraday_today)])
    print("\n🔮 Convert today’s intraday 1-min bars into a synthetic OHLC bar")
    print(df_today)
    df_combined = pd.concat([df_hist[-180:], df_today], ignore_index=True)

  # Analyze patterns + refine forecast
  results = analyze_patterns(df_combined, window=5)
  results = refine_next_predictions(results, df_combined)

  export_analysis_results(results)
  print_summary_report(results, show_forecast=True)

  print("\n🔮 Refined Forecast:")
  for day in results["next_predictions_refined"]:
    print(f"• {day}")


def fetch_intraday_bars(symbol: str, api_key: str,
    limit: int = 100) -> pd.DataFrame:
  # now = datetime.now()
  # today_str = now.strftime('%Y-%m-%d')
  now = datetime(2025, 6, 25)  # Mocked date to be commented and used real time
  today_str = now.strftime('%Y-%m-%d')
  url = (f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/"
         f"{today_str}/{today_str}?adjusted=true&sort=asc&limit={limit}&apiKey={api_key}")

  try:
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    if not data.get("results"):
      print("⚠️ No intraday data returned.")
      return None

    bars = []
    for bar in data["results"]:
      timestamp = datetime.fromtimestamp(bar["t"] / 1000)
      bars.append(
          {"Date": timestamp.strftime("%Y-%m-%d %H:%M"), "Open": bar["o"],
           "High": bar["h"], "Low": bar["l"], "Close": bar["c"],
           "Volume": bar.get("v", 0)})
    df = pd.DataFrame(bars)

    # 🔽 Filter to regular market hours only (09:30–16:00)
    df['Datetime'] = pd.to_datetime(df['Date'])
    # df = df[df['Datetime'].dt.time.between(time(9, 30), time(16, 0))]  -> this uncomment if i don't want premarket

    if not df.empty:
      print("\n🔍 Premarket or earliest intraday bars:")
      print(df.head(
          10))  # Shows earliest available data, typically starting around 04:00 ET

    return df

  except Exception as e:
    print(f"❌ Error fetching intraday bars: {e}")
    return None


if __name__ == "__main__":
  main()
