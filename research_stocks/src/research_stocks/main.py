import logging
import os
import sys

from crew import StockAnalysisCrew
from tools.run_analysis import main as run_pattern_analysis


def run_analysis_and_crew(symbol: str) -> str:
  """
  Run the pattern analysis and then the crew for the given symbol.

  Args:
      symbol: The stock symbol to analyze

  Returns:
      The final report
  """
  # First run the pattern analysis to generate the JSON file
  print(f"Running pattern analysis for {symbol}...")
  run_pattern_analysis(symbol)
  # Then run the crew to process the generated file
  print(f"Running crew analysis for {symbol}...")
  return StockAnalysisCrew().crew()


def generate_fallback_report():
  return "Analysis failed. No data available."


def safe_run(symbol: str) -> str:
  try:
    return run_analysis_and_crew(symbol)
  except Exception as e:
    logging.error(f"Critical error in execution: {e}")
    return generate_fallback_report()


def main() -> None:
  print("## Welcome to Stock Analysis Crew")
  print('-------------------------------')

  # Get symbol from user input
  symbol = input("Enter the symbol for which you want to forecast: ").strip().upper()
  if not symbol:
    symbol = "MSFT"  # Default if no input
    print(f"No symbol provided, using default: {symbol}")

  # Ensure output directory exists
  os.makedirs("output", exist_ok=True)

  # Run the analysis
  result = safe_run(symbol)

  print("\n\n########################")
  print("## Here is the Report")
  print("########################\n")
  print(result)


if __name__ == "__main__":
  main()
