# run_pattern_analysis.py
import os
from datetime import datetime

import requests
import yfinance as yf
from dotenv import load_dotenv
from polygon import RESTClient

from advanced_pattern_helpers import analyze_patterns  # ✅ new function
from advanced_pattern_helpers import export_analysis_results  # ✅ new function
from advanced_pattern_helpers import print_summary_report  # ✅ new function
from advanced_pattern_helpers import refine_next_predictions


def main():
  load_dotenv()
  # Load API key
  key = os.getenv("POLYGON_KEY")
  if not key:
    print("❌ POLYGON_KEY not set in environment variables.")
    return

  intraday_ohlc = fetch_latest_intraday_ohlc("NVDA", key)
  if intraday_ohlc:
    print("\n🕖 Premarket or Latest OHLC (1m bar):")
    print(intraday_ohlc)

  # ── Get historical data from yfinance ──
  nvda = yf.Ticker("NVDA")
  df = nvda.history(period="3mo", interval="1d")[-60:]
  df.reset_index(inplace=True)
  df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

  ohlc_data = df[['Open', 'High', 'Low', 'Close']].reset_index()
  print(ohlc_data.to_dict(orient="records"))

  # Analyze patterns
  results = analyze_patterns(df, window=1)
  results = refine_next_predictions(results, df)

  export_analysis_results(results)
  print_summary_report(results)

  print("\n🔮 Refined Forecast:")
  for day in results["next_predictions_refined"]:
    print(f"• {day}")


def fetch_latest_intraday_ohlc(symbol: str, api_key: str):
  now = datetime.now()
  today_str = now.strftime('%Y-%m-%d')
  url = (f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/"
         f"{today_str}/{today_str}?adjusted=true&sort=desc&limit=1&apiKey={api_key}")

  try:
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    if not data.get("results"):
      print(
          "⚠️ No intraday data found (possibly market closed or no premarket data yet).")
      return None

    bar = data["results"][0]
    timestamp = datetime.fromtimestamp(bar["t"] / 1000)

    return {"Date": timestamp.strftime("%Y-%m-%d %H:%M"), "Open": bar["o"],
            "High": bar["h"], "Low": bar["l"], "Close": bar["c"],
            "Volume": bar.get("v", 0)}

  except Exception as e:
    print(f"❌ Error fetching intraday data: {e}")
    return None


if __name__ == "__main__":
  main()
