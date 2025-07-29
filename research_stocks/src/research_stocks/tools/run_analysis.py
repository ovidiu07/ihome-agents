#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# run_analysis.py
# Entry-point script for the pattern-analysis tool-chain
# ---------------------------------------------------------------------------

from __future__ import annotations

import os
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

# Optional dotenv support ----------------------------------------------------
try:
  from dotenv import load_dotenv
except ImportError:  # pragma: no cover
  def load_dotenv() -> None:  # type: ignore
    print("⚠️  python-dotenv not installed ‑ set env variables manually.")

# ───────── Internal imports (adjust package path if necessary) ───────────────
from .pattern_analysis.reporting import (export_analysis_results,
                                         )


# ---------------------------------------------------------------------------


def main(symbol: str = "NVDA") -> None:
  """
  Run the complete pattern analysis pipeline for a given stock symbol.

  This function orchestrates the entire pattern analysis workflow:
  1. Fetches historical data (daily, hourly, and intraday)
  2. Analyzes patterns in the data at different time scales
  3. Filters and processes the identified patterns
  4. Generates forecasts using multiple methods (ensemble and probabilistic)
  5. Exports the results to JSON files

  Args:
      symbol: Stock ticker symbol to analyze (default: "NVDA")
              Will be automatically converted to uppercase

  Returns:
      None: Results are exported to files and printed to console

  Side Effects:
      - Creates/updates JSON files in the output directory
      - Prints analysis summaries and forecasts to the console
      - May create intraday factor snapshots
  """
  # ─── Environment / configuration ───────────────────────────────────────

  symbol = symbol.upper()
  # Save the initial results to JSON file
  results = {
    "symbol": symbol,
    "current_date": datetime.now().strftime("%Y-%m-%d")
  }

  export_analysis_results(results)


# ---------------------------------------------------------------------------
# Script entry point - allows direct execution from command line
if __name__ == "__main__":
  import sys

  # Get the stock symbol from command line argument or use NVDA as default
  input_symbol = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
  # Run the main analysis pipeline with the provided symbol
  main(input_symbol)
