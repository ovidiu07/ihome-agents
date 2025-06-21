# run_pattern_analysis.py

from advanced_pattern_helpers import analyze_patterns  # ✅ new function
from advanced_pattern_helpers import export_analysis_results  # ✅ new function
from advanced_pattern_helpers import plot_analysis_results  # ✅ new function
from advanced_pattern_helpers import print_summary_report  # ✅ new function
from test_ohlc_patterns import generate_test_ohlc


def main():
  df = generate_test_ohlc()
  results = analyze_patterns(df, window=1)

  export_analysis_results(results)
  #plot_analysis_results(results)
  print_summary_report(results)

  print("\n🧠 Pattern Recognition Results")
  if results["patterns"]:
    for p in results["patterns"]:
      print(
          f"- {p['start_date']} to {p['end_date']}: {p['pattern']} ({p['direction']}, score={p['value']})")
  else:
    print("No patterns detected.")

if __name__ == "__main__":
  main()
