import os
import json
from pathlib import Path
from tools.market_data_tools import PatternRecognitionTool

# Sample mock OHLC data designed to simulate known patterns
mock_ohlc = [
  # Bullish Engulfing (day 2 engulfs day 1)
  {"date": "2025-06-01", "open": 105, "high": 106, "low": 104, "close": 104.5},
  {"date": "2025-06-02", "open": 104, "high": 108, "low": 103.5, "close": 107.5},

  # Hammer
  {"date": "2025-06-03", "open": 107, "high": 108, "low": 104, "close": 107.8},

  # Shooting Star
  {"date": "2025-06-04", "open": 108, "high": 111, "low": 107.5, "close": 108.2},

  # Doji
  {"date": "2025-06-05", "open": 108, "high": 109, "low": 107, "close": 108},

  # Morning Star (3-day pattern)
  {"date": "2025-06-06", "open": 107.5, "high": 108, "low": 106.5, "close": 107},
  {"date": "2025-06-07", "open": 106.5, "high": 107, "low": 104.5, "close": 105},
  {"date": "2025-06-08", "open": 105.5, "high": 108.5, "low": 105, "close": 108},

  # Fill remaining to satisfy >10 bars for pattern recognition
  *[
    {"date": f"2025-06-{day:02d}", "open": 108, "high": 109, "low": 107, "close": 108}
    for day in range(9, 21)
  ]
]

def test_pattern_recognition_tool():
  tool = PatternRecognitionTool()
  ticker = "TEST"

  # Run the pattern recognition tool
  results = tool._run(ticker=ticker, ohlc_data=mock_ohlc)

  # Check return type
  assert isinstance(results, list), "Expected a list of detected patterns"

  # Check markdown output
  md_file = Path(f"{ticker.upper()}_patterns.md")
  assert md_file.exists(), f"Markdown file not found: {md_file}"
  print(f"✅ {md_file} exists")

  with open(md_file, "r") as f:
    md_content = f.read()
    print("📄 Markdown preview:\n", md_content[:300], "...")

  # Check JSON output
  json_file = Path(f"{ticker.upper()}_patterns.json")
  assert json_file.exists(), f"JSON file not found: {json_file}"
  print(f"✅ {json_file} exists")

  with open(json_file, "r") as f:
    json_data = json.load(f)
    assert isinstance(json_data, list), "Expected a list in JSON output"
    if json_data:
      print(f"🔍 Found {len(json_data)} pattern(s):")
      for p in json_data[:5]:
        print(" -", p)

if __name__ == "__main__":
  test_pattern_recognition_tool()