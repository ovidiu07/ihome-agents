# Pattern Analysis Package

This package contains modules for stock pattern analysis, including data fetching, pattern detection, filtering, forecasting, and reporting.

## Overview

The Pattern Analysis package is a modular implementation of stock pattern analysis tools. It provides functionality for:

- Fetching stock market data from various sources
- Detecting candlestick and chart patterns
- Filtering and processing detected patterns
- Forecasting based on pattern analysis
- Reporting and visualization of analysis results
- Backtesting pattern-based trading strategies

## Module Structure

The package is organized into the following modules:

### `data_fetchers.py`
Functions for fetching stock market data from various sources:
- `fetch_intraday_bars`: Fetches intraday data from Polygon.io
- `fetch_daily_history`: Fetches daily historical data using yfinance

### `utils.py`
Utility functions for pattern analysis:
- `get_pattern_reliability`: Returns reliability scores for patterns
- `ensure_pattern_dates_are_datetime`: Ensures dates are in datetime format
- `slope`: Calculates the slope of a series

### `candlestick_patterns.py`
Functions for detecting candlestick patterns:
- `cs_hammer`, `cs_doji`, etc.: Individual pattern detection functions
- `detect_candlestick_patterns`: Main function for detecting all candlestick patterns

### `chart_patterns.py`
Functions for detecting chart patterns:
- `detect_pivots`: Detects pivot points in price data
- `detect_head_shoulders_pivot`: Detects head and shoulders patterns
- `detect_double_tops_bottoms_pivot`: Detects double top/bottom patterns
- `detect_triangle_pivot`: Detects triangle patterns

### `pattern_analyzer.py`
Core pattern analysis functions:
- `calculate_pattern_score`: Calculates scores for patterns
- `resolve_conflicts`: Resolves conflicts between overlapping patterns
- `analyze_patterns`: Main function for analyzing patterns

### `pattern_filters.py`
Functions for filtering and processing patterns:
- `drop_duplicates`: Removes duplicate patterns
- `suppress_nearby_hits`: Removes patterns that are too close to each other
- `cluster_and_keep_best`: Collapses overlapping patterns
- `filter_patterns_by_criteria`: Filters patterns based on criteria

### `forecast_utils.py`
Functions for forecasting and bias calculation:
- `get_intraday_bias`: Calculates bias from intraday patterns
- `get_daily_bias`: Calculates bias from daily patterns
- `blended_forecast`: Creates an ensemble forecast
- `calculate_vwap_obv_trend`: Calculates VWAP and OBV trend
- `calculate_atr`: Calculates Average True Range

### `forecasting.py`
Functions for forecasting based on pattern analysis:
- `refine_next_predictions`: Refines predictions for the next period
- `build_feature_stack`: Builds features for forecasting
- `probabilistic_day_forecast`: Generates probabilistic forecasts

### `reporting.py`
Functions for reporting and visualization:
- `export_analysis_results`: Exports results to JSON
- `print_summary_report`: Prints a summary report
- `generate_evolving_daily_ohlc`: Generates daily OHLC from intraday data

### `backtesting.py`
Functions for backtesting pattern strategies:
- `backtest_pattern_strategy`: Backtests a pattern-based trading strategy

## Usage

To use the Pattern Analysis package, import the required modules and functions:

```python
# Import data fetching functions
from research_stocks.tools.pattern_analysis.data_fetchers import fetch_daily_history, fetch_intraday_bars

# Import pattern analysis functions
from research_stocks.tools.pattern_analysis.pattern_analyzer import analyze_patterns

# Import forecasting functions
from research_stocks.tools.pattern_analysis.forecasting import refine_next_predictions, probabilistic_day_forecast

# Example usage
df = fetch_daily_history("AAPL", period="1y")
results = analyze_patterns(df)
results = refine_next_predictions(results, df)
```

For a complete example, see the `run_analysis.py` script in the parent directory.
