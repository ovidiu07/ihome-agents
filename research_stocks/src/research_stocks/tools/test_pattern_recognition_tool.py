import os
import json
from pathlib import Path
from tools.market_data_tools import PatternRecognitionTool

mock_ohlc = [
  # Engulfing (bullish)
  {"date": "2025-06-01", "open": 106, "high": 107, "low": 105.8, "close": 105.7},
  {"date": "2025-06-02", "open": 105.6, "high": 109.0, "low": 105.5, "close": 108.9},

  # Hammer
  {"date": "2025-06-03", "open": 109.0, "high": 109.2, "low": 105.0, "close": 109.1},

  # Shooting Star
  {"date": "2025-06-04", "open": 109.0, "high": 112.5, "low": 108.9, "close": 109.1},

  # Doji
  {"date": "2025-06-05", "open": 108.8, "high": 109.2, "low": 108.6, "close": 108.8},

  # Morning Star: red, small body, green
  {"date": "2025-06-06", "open": 109.0, "high": 109.1, "low": 105.8, "close": 106.0},
  {"date": "2025-06-07", "open": 106.0, "high": 106.5, "low": 105.0, "close": 105.8},
  {"date": "2025-06-08", "open": 106.0, "high": 109.5, "low": 105.8, "close": 109.3},

  # Filler bars to reach 20
  *[
    {"date": f"2025-06-{day:02d}", "open": 110, "high": 111, "low": 109, "close": 110}
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