# test_ohlc_patterns.py

import pandas as pd


def generate_test_ohlc():
  return pd.DataFrame(
    [
          {"Date": "2025-05-29", "Open": 117.20, "High": 118.70, "Low": 115.50, "Close": 116.88},
          {"Date": "2025-05-30", "Open": 116.88, "High": 118.22, "Low": 115.22, "Close": 115.80},
          {"Date": "2025-05-31", "Open": 115.80, "High": 116.50, "Low": 114.10, "Close": 114.31},
          {"Date": "2025-06-01", "Open": 114.31, "High": 116.80, "Low": 113.50, "Close": 116.24},
          {"Date": "2025-06-02", "Open": 116.24, "High": 116.50, "Low": 114.88, "Close": 115.16},
          {"Date": "2025-06-03", "Open": 115.16, "High": 116.00, "Low": 114.20, "Close": 114.89},
          {"Date": "2025-06-04", "Open": 114.89, "High": 116.50, "Low": 113.80, "Close": 115.55},
          {"Date": "2025-06-05", "Open": 115.55, "High": 118.00, "Low": 114.95, "Close": 117.28},
          {"Date": "2025-06-06", "Open": 117.28, "High": 118.75, "Low": 116.40, "Close": 117.98},
          {"Date": "2025-06-09", "Open": 117.98, "High": 119.60, "Low": 117.50, "Close": 118.77},
          {"Date": "2025-06-10", "Open": 118.77, "High": 120.15, "Low": 118.00, "Close": 119.64},
          {"Date": "2025-06-11", "Open": 119.64, "High": 121.20, "Low": 119.10, "Close": 120.95},
          {"Date": "2025-06-12", "Open": 120.95, "High": 123.00, "Low": 120.20, "Close": 122.35},
          {"Date": "2025-06-13", "Open": 122.35, "High": 124.00, "Low": 121.80, "Close": 123.75},
          {"Date": "2025-06-14", "Open": 124.25, "High": 125.80, "Low": 123.90, "Close": 125.33},
          {"Date": "2025-06-17", "Open": 125.33, "High": 127.10, "Low": 124.70, "Close": 126.44},
          {"Date": "2025-06-18", "Open": 126.44, "High": 127.80, "Low": 125.10, "Close": 126.75},
          {"Date": "2025-06-19", "Open": 127.20, "High": 128.90, "Low": 126.20, "Close": 127.95},
          {"Date": "2025-06-20", "Open": 128.80, "High": 129.90, "Low": 127.30, "Close": 128.75},
          {"Date": "2025-06-21", "Open": 129.45, "High": 131.12, "Low": 128.88, "Close": 130.65}

    ])


# 🔮 Forecast for Next 2 Days NVDA:
#         - 2025-06-23 to 2025-06-23: Bullish Continuation (Confidence: Moderate)
# OHLC Forecast:
# • 2025-06-23: O=143.85 H=144.50 L=142.90 C=143.75
#
# 🔮 Refined Forecast:
# • {'Open': np.float64(143.85), 'High': 145.33, 'Low': 142.38, 'Close': 143.87}


#
# 🔮 Forecast for Next 2 Days TESLA:
#         - 2025-06-23 to 2025-06-23: Bullish Continuation (Confidence: Moderate)
# OHLC Forecast:
# • 2025-06-23: O=322.15 H=328.45 L=319.15 C=325.05
#
# 🔮 Refined Forecast:
# • {'Open': np.float64(322.16), 'High': 329.09, 'Low': 315.76, 'Close': 322.8}
