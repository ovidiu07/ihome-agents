import logging
import sys

from crew import StockAnalysisCrew


def run(symbol: str) -> str:
  query = input("Enter the company you want to forecast for: ").strip()

  inputs = {'query': query, 'symbol': symbol}
  return StockAnalysisCrew().crew().kickoff(inputs=inputs)


def generate_fallback_report():
  return "Crew execution failed. No data available."


def safe_run(symbol: str) -> str:
  try:
    return run(symbol)
  except Exception as e:
    logging.error(f"Critical error in crew execution: {e}")
    return generate_fallback_report()


def main(symbol: str = "MSFT") -> None:
  symbol = symbol.upper()
  print("## Welcome to Stock Analysis Crew")
  print('-------------------------------')
  result = safe_run(symbol)
  print("\n\n########################")
  print("## Here is the Report")
  print("########################\n")
  print(result)


if __name__ == "__main__":
  input_symbol = sys.argv[1] if len(sys.argv) > 1 else "MSFT"
  main(input_symbol)
