# run_pattern_analysis.py
from advanced_pattern_helpers import refine_next_predictions
from advanced_pattern_helpers import analyze_patterns  # ✅ new function
from advanced_pattern_helpers import export_analysis_results  # ✅ new function
from advanced_pattern_helpers import print_summary_report  # ✅ new function
from test_ohlc_patterns import generate_test_ohlc
import yfinance as yf



def main():
  # Download latest 20 daily OHLC for NVDA
  nvda = yf.Ticker("TSLA")
  df = nvda.history(period="1mo", interval="1d")[-20:]
  df.reset_index(inplace=True)
  df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')  # Optional for consistency
  # Format for JSON
  ohlc_data = df[['Open', 'High', 'Low', 'Close']].reset_index()
  print(ohlc_data.to_dict(orient="records"))
  results = analyze_patterns(df, window=1)
  results = refine_next_predictions(results, df)   # adds “next_predictions_refined”


  export_analysis_results(results)
  #plot_analysis_results(results)
  print_summary_report(results)

  print("\n🔮 Refined Forecast:")
  for day in results["next_predictions_refined"]:
    print(f"• {day}")
  # print("\n🧠 Pattern Recognition Results")
  # if results["patterns"]:
  #   for p in results["patterns"]:
  #     print(
  #         f"- {p['start_date']} to {p['end_date']}: {p['pattern']} ({p['direction']}, score={p['value']})")
  # else:
  #   print("No patterns detected.")

if __name__ == "__main__":
  main()
