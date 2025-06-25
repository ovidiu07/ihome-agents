# test_ohlc_patterns.py

import pandas as pd


def generate_test_ohlc():
  return pd.DataFrame(
    [
            {"Date": "2025-06-23", "Open": 142.50, "High": 144.78, "Low": 142.03, "Close": 144.22},
            {"Date": "2025-06-20", "Open": 145.45, "High": 146.20, "Low": 142.65, "Close": 143.85},
            {"Date": "2025-06-18", "Open": 144.01, "High": 145.65, "Low": 143.12, "Close": 145.48},
            {"Date": "2025-06-17", "Open": 144.49, "High": 145.22, "Low": 143.78, "Close": 144.12},
            {"Date": "2025-06-16", "Open": 143.35, "High": 146.18, "Low": 143.20, "Close": 144.69},
            {"Date": "2025-06-13", "Open": 142.48, "High": 143.58, "Low": 140.86, "Close": 141.97},
            {"Date": "2025-06-12", "Open": 141.97, "High": 145.00, "Low": 141.85, "Close": 145.00},
            {"Date": "2025-06-11", "Open": 144.61, "High": 144.99, "Low": 141.87, "Close": 142.83},
            {"Date": "2025-06-10", "Open": 142.69, "High": 144.29, "Low": 141.53, "Close": 143.96},
            {"Date": "2025-06-09", "Open": 143.19, "High": 145.00, "Low": 141.94, "Close": 142.63},
            {"Date": "2025-06-06", "Open": 142.51, "High": 143.27, "Low": 141.51, "Close": 141.72},
            {"Date": "2025-06-05", "Open": 142.17, "High": 144.00, "Low": 138.83, "Close": 139.99},
            {"Date": "2025-06-04", "Open": 142.19, "High": 142.39, "Low": 139.55, "Close": 141.92},
            {"Date": "2025-06-03", "Open": 138.78, "High": 142.00, "Low": 137.95, "Close": 141.22},
            {"Date": "2025-06-02", "Open": 135.49, "High": 138.12, "Low": 135.40, "Close": 137.38},
            {"Date": "2025-05-30", "Open": 138.72, "High": 139.62, "Low": 132.92, "Close": 135.13},
            {"Date": "2025-05-29", "Open": 142.25, "High": 143.49, "Low": 137.91, "Close": 139.19},
            {"Date": "2025-05-28", "Open": 136.03, "High": 137.25, "Low": 134.79, "Close": 134.81},
            {"Date": "2025-05-27", "Open": 134.15, "High": 135.66, "Low": 133.31, "Close": 135.50},
            {"Date": "2025-05-23", "Open": 130.00, "High": 132.68, "Low": 129.16, "Close": 131.29},
            {"Date": "2025-05-22", "Open": 132.23, "High": 134.25, "Low": 131.55, "Close": 132.83},
            {"Date": "2025-05-21", "Open": 133.06, "High": 137.40, "Low": 130.59, "Close": 131.80},
            {"Date": "2025-05-20", "Open": 134.29, "High": 134.58, "Low": 132.62, "Close": 134.38},
            {"Date": "2025-05-19", "Open": 132.39, "High": 135.87, "Low": 132.39, "Close": 135.57},
            {"Date": "2025-05-16", "Open": 136.22, "High": 136.35, "Low": 133.46, "Close": 135.40},
            {"Date": "2025-05-15", "Open": 134.30, "High": 136.30, "Low": 132.66, "Close": 134.83},
            {"Date": "2025-05-14", "Open": 133.20, "High": 135.44, "Low": 131.68, "Close": 135.34},
            {"Date": "2025-05-13", "Open": 124.98, "High": 131.22, "Low": 124.47, "Close": 129.93},
            {"Date": "2025-05-12", "Open": 121.97, "High": 123.00, "Low": 120.28, "Close": 123.00},
            {"Date": "2025-05-09", "Open": 117.35, "High": 118.23, "Low": 115.21, "Close": 116.65}

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
