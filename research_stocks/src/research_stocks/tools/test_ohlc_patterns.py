# test_ohlc_patterns.py

import pandas as pd


def generate_test_ohlc():
  return pd.DataFrame([# Head and Shoulders
    {"Date": "2025-06-01", "Open": 100, "High": 108, "Low": 99, "Close": 105},
    {"Date": "2025-06-02", "Open": 105, "High": 110, "Low": 103, "Close": 107},
    {"Date": "2025-06-03", "Open": 107, "High": 106, "Low": 104, "Close": 105},

    # Inverse Head and Shoulders
    {"Date": "2025-06-04", "Open": 105, "High": 106, "Low": 100, "Close": 104},
    {"Date": "2025-06-05", "Open": 104, "High": 105, "Low": 98, "Close": 102},
    {"Date": "2025-06-06", "Open": 102, "High": 104, "Low": 97, "Close": 103},

    # Ascending Triangle
    {"Date": "2025-06-07", "Open": 103, "High": 108, "Low": 100, "Close": 107},
    {"Date": "2025-06-08", "Open": 107, "High": 109, "Low": 105, "Close": 108},
    {"Date": "2025-06-09", "Open": 108, "High": 110, "Low": 106, "Close": 109},

    # Double bottom
    {"Date": "2025-06-10", "Open": 110, "High": 112, "Low": 105, "Close": 106},
    {"Date": "2025-06-11", "Open": 106, "High": 108, "Low": 105, "Close": 107},
    {"Date": "2025-06-12", "Open": 107, "High": 107, "Low": 104, "Close": 105},
    {"Date": "2025-06-13", "Open": 105, "High": 110, "Low": 105, "Close": 109},
    {"Date": "2025-06-14", "Open": 109, "High": 112, "Low": 108, "Close": 111}, ])

    # # Double Top
    # {"Date": "2025-06-13", "Open": 102, "High": 110, "Low": 101, "Close": 108},
    # {"Date": "2025-06-14", "Open": 108, "High": 111, "Low": 102, "Close": 109},
    # {"Date": "2025-06-15", "Open": 109, "High": 109, "Low": 104, "Close": 106},
    #
    # # Double Bottom
    # {"Date": "2025-06-16", "Open": 106, "High": 107, "Low": 99, "Close": 101},
    # {"Date": "2025-06-17", "Open": 101, "High": 105, "Low": 98, "Close": 103},
    # {"Date": "2025-06-18", "Open": 103, "High": 104, "Low": 97, "Close": 102},
    #
    # # Wedge Up
    # {"Date": "2025-06-19", "Open": 102, "High": 106, "Low": 100, "Close": 105},
    # {"Date": "2025-06-20", "Open": 105, "High": 107, "Low": 103, "Close": 106},
    # {"Date": "2025-06-21", "Open": 106, "High": 109, "Low": 104, "Close": 108},
    #
    # # Wedge Down
    # {"Date": "2025-06-22", "Open": 108, "High": 107, "Low": 103, "Close": 104},
    # {"Date": "2025-06-23", "Open": 104, "High": 106, "Low": 102, "Close": 103},
    # {"Date": "2025-06-24", "Open": 103, "High": 105, "Low": 101, "Close": 101},
    #
    # # Channel Up
    # {"Date": "2025-06-25", "Open": 101, "High": 106, "Low": 100, "Close": 104},
    # {"Date": "2025-06-26", "Open": 104, "High": 108, "Low": 102, "Close": 106},
    # {"Date": "2025-06-27", "Open": 106, "High": 110, "Low": 104, "Close": 108},
    #
    # # Channel Down
    # {"Date": "2025-06-28", "Open": 108, "High": 109, "Low": 104, "Close": 106},
    # {"Date": "2025-06-29", "Open": 106, "High": 107, "Low": 103, "Close": 104},
    # {"Date": "2025-06-30", "Open": 104, "High": 105, "Low": 100, "Close": 102},
    #
    # # Pivot points (higher high, lower low, etc.)
    # {"Date": "2025-07-01", "Open": 102, "High": 110, "Low": 100, "Close": 108},
    # {"Date": "2025-07-02", "Open": 108, "High": 112, "Low": 105, "Close": 111},
    # {"Date": "2025-07-03", "Open": 111, "High": 113, "Low": 108,
    #  "Close": 110}, ])

