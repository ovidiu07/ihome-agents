# tools/test_forecast_signal_tool.py

from tools.market_data_tools import ForecastSignalTool

if __name__ == "__main__":
  # Example close prices (should be at least 35 values)
  close_prices = [
    478.25, 476.89, 475.11, 473.92, 471.56, 469.02, 468.79, 467.55, 466.48, 464.12,
    462.89, 461.03, 460.15, 458.66, 457.28, 455.77, 454.20, 452.76, 450.10, 448.97,
    447.68, 446.50, 444.88, 443.20, 441.60, 440.25, 439.10, 437.55, 436.88, 435.92,
    434.45, 433.80, 432.10, 430.75, 429.88
  ]

  tool = ForecastSignalTool()
  result = tool.run(ticker="SPY", close_prices=close_prices, vix=18.3)

  print("=== Forecast Signal Result ===")
  for key, value in result.items():
    print(f"- {key}: {value}")