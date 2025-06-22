# test_ohlc_patterns.py

import pandas as pd


def generate_test_ohlc():
  return pd.DataFrame([# Head and Shoulders
    [
      {"Date": "2025-06-03", "Open": 114.02, "High": 116.55, "Low": 113.12, "Close": 115.80},
      {"Date": "2025-06-04", "Open": 116.03, "High": 117.18, "Low": 113.91, "Close": 114.31},
      {"Date": "2025-06-05", "Open": 114.70, "High": 116.92, "Low": 114.18, "Close": 116.24},
      {"Date": "2025-06-06", "Open": 116.68, "High": 117.30, "Low": 114.62, "Close": 115.16},
      {"Date": "2025-06-07", "Open": 115.40, "High": 116.05, "Low": 113.95, "Close": 114.89},
      {"Date": "2025-06-10", "Open": 115.22, "High": 118.74, "Low": 114.57, "Close": 117.98},
      {"Date": "2025-06-11", "Open": 118.10, "High": 119.66, "Low": 117.13, "Close": 118.77},
      {"Date": "2025-06-12", "Open": 119.20, "High": 121.35, "Low": 118.30, "Close": 120.50},
      {"Date": "2025-06-13", "Open": 121.05, "High": 123.60, "Low": 120.88, "Close": 122.74},
      {"Date": "2025-06-14", "Open": 123.00, "High": 124.20, "Low": 122.02, "Close": 123.87}
    ], ])

