# Stock Market Pattern Analysis System Documentation

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Pattern Detection Algorithms](#pattern-detection-algorithms)
4. [Backtesting Framework](#backtesting-framework)
5. [Prediction System](#prediction-system)
6. [Output Formats](#output-formats)
7. [Usage Examples](#usage-examples)
8. [API Reference](#api-reference)

## Overview

The Stock Market Pattern Analysis System is a comprehensive tool for analyzing stock price patterns, backtesting trading strategies, and making predictions about future price movements. The system uses advanced pattern recognition algorithms to identify common chart patterns such as Head and Shoulders, Double Tops/Bottoms, Triangles, and Wedges.

Key features of the system include:
- Detection of multiple chart patterns with configurable parameters
- Backtesting of trading strategies based on detected patterns
- Prediction of future price movements
- Export of results to various formats (CSV, JSON)
- Visualization of results with plots

## System Architecture

The system consists of several components that work together to provide a complete pattern analysis solution:

```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│                     │     │                     │     │                     │
│   Input Data        │────▶│   Pattern Analysis  │────▶│   Results Output    │
│   (OHLC Data)       │     │   Engine            │     │   (JSON, CSV)       │
│                     │     │                     │     │                     │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘
                                      │
                                      │
                                      ▼
                            ┌─────────────────────┐
                            │                     │
                            │   Visualization     │
                            │   & Reporting       │
                            │                     │
                            └─────────────────────┘
```

### Workflow

The typical workflow for using the system is as follows:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │     │                 │
│  1. Load Data   │────▶│  2. Detect      │────▶│  3. Backtest    │────▶│  4. Generate    │
│                 │     │     Patterns    │     │     Strategy    │     │     Predictions │
│                 │     │                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
                                                                               │
                                                                               │
                                                                               ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │     │                 │
│  8. Make        │◀────│  7. Visualize   │◀────│  6. Generate    │◀────│  5. Export      │
│     Decisions   │     │     Results     │     │     Reports     │     │     Results     │
│                 │     │                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
```

1. **Load Data**: Load OHLC (Open, High, Low, Close) data from a CSV file, API, or other source.
2. **Detect Patterns**: Use the pattern detection algorithms to identify chart patterns in the data.
3. **Backtest Strategy**: Backtest a trading strategy based on the detected patterns.
4. **Generate Predictions**: Generate predictions for future price movements based on the detected patterns.
5. **Export Results**: Export the results to CSV and JSON files for further analysis.
6. **Generate Reports**: Generate summary reports of the analysis results.
7. **Visualize Results**: Create visualizations of the results, such as equity curves and pattern charts.
8. **Make Decisions**: Use the analysis results to make trading decisions.

### Components

1. **Input Data**: The system accepts OHLC (Open, High, Low, Close) data in the form of a pandas DataFrame. This data can be loaded from various sources, including CSV files, APIs, or generated synthetically for testing.

   #### Data Format Requirements

   The input data must be a pandas DataFrame with the following columns:
   - `Date`: Date of the price data (string or datetime)
   - `Open`: Opening price (float)
   - `High`: Highest price (float)
   - `Low`: Lowest price (float)
   - `Close`: Closing price (float)
   - `Volume`: Trading volume (optional, float)

   Example of a valid input DataFrame:

   ```
   |    Date    | Open  | High  | Low   | Close | Volume  |
   |------------|-------|-------|-------|-------|---------|
   | 2025-05-01 | 108.45| 110.70| 107.50| 109.88| 1000000 |
   | 2025-05-02 | 110.00| 111.30| 109.10| 110.95| 1200000 |
   | 2025-05-05 | 111.20| 112.90| 110.60| 112.35| 900000  |
   | ...        | ...   | ...   | ...   | ...   | ...     |
   ```

2. **Pattern Analysis Engine**: The core of the system, which analyzes the input data to detect patterns, backtest strategies, and make predictions. This component is implemented in the `advanced_pattern_helpers.py` file.

3. **Results Output**: The system exports the results of the analysis to various formats, including CSV and JSON files. These files are stored in the `output` directory.

4. **Visualization & Reporting**: The system provides functions for visualizing the results and generating reports. This includes plotting equity curves and printing summary reports.

## Pattern Detection Algorithms

The system implements several pattern detection algorithms, each designed to identify specific chart patterns. These algorithms use a combination of pivot point analysis, slope calculation, and other technical indicators to identify patterns.

### Pivot Point Detection

Pivot points are local highs and lows in the price data. The system uses a configurable window to identify these points, which are then used as the basis for pattern detection.

```
    High
     │
     │    ┌───┐
     │    │   │
     │┌───┘   └───┐
     ││           │
     │└───┐   ┌───┘
     │    │   │
     │    └───┘
     │
     └────────────────▶ Time
```

### Head and Shoulders Pattern

The Head and Shoulders pattern consists of three peaks, with the middle peak (head) higher than the two outer peaks (shoulders). The system identifies this pattern by analyzing the relationship between pivot points.

```
    Price
     │
     │      Head
     │       │
     │       ┌─┐
     │       │ │
     │ Left  │ │  Right
     │Shoulder │ Shoulder
     │  ┌─┐   │ │   ┌─┐
     │  │ │   │ │   │ │
     │  │ │   │ │   │ │
     │  │ └───┘ └───┘ │
     │  │             │
     │  └─────────────┘
     │
     └────────────────────▶ Time
```

### Double Top/Bottom Pattern

The Double Top pattern consists of two peaks at approximately the same price level, separated by a trough. The Double Bottom pattern is the inverse, with two troughs at approximately the same level, separated by a peak.

```
    Price
     │
     │  Double Top
     │    ┌─┐ ┌─┐
     │    │ │ │ │
     │    │ │ │ │
     │    │ └─┘ │
     │    │     │
     │    └─────┘
     │
     │  Double Bottom
     │    ┌─────┐
     │    │     │
     │    │ ┌─┐ │
     │    │ │ │ │
     │    │ │ │ │
     │    └─┘ └─┘
     │
     └────────────────────▶ Time
```

### Triangle Patterns

Triangle patterns are characterized by converging trendlines. The system identifies ascending triangles (flat top, rising bottom), descending triangles (flat bottom, falling top), and symmetrical triangles (both trendlines converging).

```
    Price
     │
     │  Ascending Triangle
     │    ─────────────
     │    │           /
     │    │         /
     │    │       /
     │    │     /
     │    │   /
     │    │ /
     │    │/
     │
     └────────────────────▶ Time
```

### Wedge Patterns

Wedge patterns are similar to triangles but both trendlines move in the same direction. The system identifies rising wedges (bearish) and falling wedges (bullish).

```
    Price
     │
     │  Rising Wedge (Bearish)
     │         /│
     │       /  │
     │     /    │
     │   /      │
     │ /        │
     │/         │
     │
     │  Falling Wedge (Bullish)
     │  │\
     │  │  \
     │  │    \
     │  │      \
     │  │        \
     │  │          \
     │
     └────────────────────▶ Time
```

### Channel Patterns

Channel patterns consist of parallel trendlines. The system identifies up channels (bullish) and down channels (bearish).

```
    Price
     │
     │  Channel Up (Bullish)
     │    /       /
     │   /       /
     │  /       /
     │ /       /
     │/       /
     │
     │  Channel Down (Bearish)
     │  \       \
     │   \       \
     │    \       \
     │     \       \
     │      \       \
     │
     └────────────────────▶ Time
```

### Engulfing Patterns

Engulfing patterns are candlestick patterns where one candle completely engulfs the previous candle. The system identifies bullish engulfing patterns (bullish) and bearish engulfing patterns (bearish).

```
    Price
     │
     │  Bullish Engulfing
     │    │
     │    │  ┌───┐
     │    │  │   │
     │    │  │   │
     │    │  │   │
     │    │  │   │
     │    │  │   │
     │    └──┘   │
     │           │
     │           └───
     │
     │  Bearish Engulfing
     │           ┌───
     │           │
     │    ┌──┐   │
     │    │  │   │
     │    │  │   │
     │    │  │   │
     │    │  │   │
     │    │  │   │
     │    │  └───┘
     │    │
     │
     └────────────────────▶ Time
```

## Backtesting Framework

The system includes a simple backtesting framework that allows users to evaluate the performance of trading strategies based on detected patterns. The backtesting process involves:

1. **Signal Generation**: The system generates trading signals based on detected patterns.
2. **Position Sizing**: The system calculates the appropriate position size based on the risk parameter.
3. **Entry and Exit**: The system simulates entering and exiting positions based on the signals.
4. **Performance Tracking**: The system tracks the equity curve and calculates performance metrics.

The backtesting results are stored in the `equity_curve.csv` and `equity_curve.json` files, and a summary is stored in the `backtest_summary.json` file.

## Prediction System

The system includes a prediction component that forecasts future price movements based on detected patterns. The prediction process involves:

1. **Pattern Analysis**: The system analyzes recent patterns to determine the likely direction of future price movements.
2. **Confidence Calculation**: The system calculates a confidence level for the prediction based on the strength of the patterns.
3. **OHLC Forecast**: The system generates forecasted OHLC values for future days.

The prediction results are stored in the `next_predictions.csv` and `next_predictions.json` files.

## Output Formats

The system exports the results of the analysis to various formats in the `output` directory:

```
output/
├── backtest_summary.json     # Summary of backtest results
├── detected_patterns.csv     # Detected patterns in CSV format
├── detected_patterns.json    # Detected patterns in JSON format
├── equity_curve.csv          # Equity curve data in CSV format
├── equity_curve.json         # Equity curve data in JSON format
├── next_predictions.csv      # Predictions in CSV format
├── next_predictions.json     # Predictions in JSON format
└── results_cache.json        # Cache of all results
```

### JSON Files

- `detected_patterns.json`: Contains information about detected patterns, including the pattern type, date range, direction, and confidence.
- `equity_curve.json`: Contains the equity curve data from the backtest.
- `backtest_summary.json`: Contains a summary of the backtest results, including start equity, end equity, and net PnL.
- `next_predictions.json`: Contains predictions for future price movements, including the expected pattern, confidence, and forecasted OHLC values.
- `results_cache.json`: Contains a cache of all results for quick reuse.

### CSV Files

- `detected_patterns.csv`: CSV version of the detected patterns.
- `equity_curve.csv`: CSV version of the equity curve data.
- `next_predictions.csv`: CSV version of the predictions.

## Usage Examples

### Basic Usage

```python
# Import the necessary functions
from research_stocks.tools.advanced_pattern_helpers import analyze_patterns, export_analysis_results, print_summary_report
import pandas as pd

# Load OHLC data
df = pd.read_csv('data.csv')

# Analyze patterns
results = analyze_patterns(df, window=5)

# Export results
export_analysis_results(results, output_dir='output')

# Print summary report
print_summary_report(results)
```

### Visualization

```python
# Import the necessary functions
from research_stocks.tools.advanced_pattern_helpers import analyze_patterns, plot_analysis_results
import pandas as pd

# Load OHLC data
df = pd.read_csv('data.csv')

# Analyze patterns
results = analyze_patterns(df, window=5)

# Plot results
plot_analysis_results(results)
```

### Custom Pattern Detection

```python
# Import the necessary functions
from research_stocks.tools.advanced_pattern_helpers import detect_head_shoulders_pivot, detect_pivots
import pandas as pd

# Load OHLC data
df = pd.read_csv('data.csv')

# Detect pivots
pivots = detect_pivots(df)

# Detect head and shoulders patterns
patterns = detect_head_shoulders_pivot(df, pivots)

# Print detected patterns
for pattern in patterns:
    print(f"{pattern['pattern']} detected from {pattern['start_date']} to {pattern['end_date']}")
```

## API Reference

### Core Functions

#### `analyze_patterns(df, window=5, volume_col=None)`

Analyzes the input data to detect patterns, backtest strategies, and make predictions.

- `df`: pandas DataFrame containing OHLC data
- `window`: Window size for pattern detection
- `volume_col`: Name of the volume column (optional)

Returns a dictionary containing:
- `patterns`: List of detected patterns
- `equity_curve`: Equity curve data from the backtest
- `backtest_summary`: Summary of the backtest results
- `next_predictions`: Predictions for future price movements

#### `export_analysis_results(results, output_dir='output')`

Exports the results of the analysis to various formats.

- `results`: Results dictionary from `analyze_patterns`
- `output_dir`: Directory to store the output files

#### `plot_analysis_results(results)`

Plots the results of the analysis.

- `results`: Results dictionary from `analyze_patterns`

#### `print_summary_report(results)`

Prints a summary report of the analysis results.

- `results`: Results dictionary from `analyze_patterns`

### Pattern Detection Functions

#### `detect_pivots(df, left=3, right=3, min_diff=0.005, tolerate_equal=True)`

Detects pivot points in the price data.

- `df`: pandas DataFrame containing OHLC data
- `left`: Number of bars to look back
- `right`: Number of bars to look forward
- `min_diff`: Minimum price difference to consider a pivot
- `tolerate_equal`: Whether to allow equality on one side

Returns a pandas DataFrame containing pivot points.

#### `detect_head_shoulders_pivot(df, pivots, volume_col=None)`

Detects head and shoulders patterns.

- `df`: pandas DataFrame containing OHLC data
- `pivots`: Pivot points DataFrame from `detect_pivots`
- `volume_col`: Name of the volume column (optional)

Returns a list of detected patterns.

#### `detect_double_tops_bottoms_pivot(df, pivots)`

Detects double top and double bottom patterns.

- `df`: pandas DataFrame containing OHLC data
- `pivots`: Pivot points DataFrame from `detect_pivots`

Returns a list of detected patterns.

#### `detect_triangle_pivot(df, pivots, window=10)`

Detects triangle patterns.

- `df`: pandas DataFrame containing OHLC data
- `pivots`: Pivot points DataFrame from `detect_pivots`
- `window`: Window size for pattern detection

Returns a list of detected patterns.

#### `detect_wedge_patterns_slope(df, window=10)`

Detects wedge patterns.

- `df`: pandas DataFrame containing OHLC data
- `window`: Window size for pattern detection

Returns a list of detected patterns.

#### `detect_channel_patterns_slope(df, window=10)`

Detects channel patterns.

- `df`: pandas DataFrame containing OHLC data
- `window`: Window size for pattern detection

Returns a list of detected patterns.

#### `detect_engulfing_patterns(df)`

Detects engulfing patterns.

- `df`: pandas DataFrame containing OHLC data

Returns a list of detected patterns.

### Backtesting Functions

#### `backtest_pattern_strategy(df, patterns, risk=0.01)`

Backtests a trading strategy based on detected patterns.

- `df`: pandas DataFrame containing OHLC data
- `patterns`: List of detected patterns
- `risk`: Risk parameter for position sizing

Returns a pandas DataFrame containing the equity curve.

### Utility Functions

#### `slope(series)`

Calculates the slope of a time series.

- `series`: pandas Series

Returns the slope as a float.

#### `calculate_pattern_score(pattern, df, volume_col=None)`

Calculates a score for a detected pattern.

- `pattern`: Pattern dictionary
- `df`: pandas DataFrame containing OHLC data
- `volume_col`: Name of the volume column (optional)

Returns the score as a float.

#### `resolve_conflicts(patterns)`

Resolves conflicts between overlapping patterns.

- `patterns`: List of detected patterns

Returns a list of non-conflicting patterns.

## Conclusion

The Stock Market Pattern Analysis System is a powerful tool for analyzing stock price patterns, backtesting trading strategies, and making predictions about future price movements. By following the workflow described in this documentation, users can leverage the system to gain insights into market behavior and make more informed trading decisions.

Key takeaways:
- The system can detect multiple chart patterns, including Head and Shoulders, Double Tops/Bottoms, Triangles, and Wedges.
- The backtesting framework allows users to evaluate the performance of trading strategies based on detected patterns.
- The prediction system forecasts future price movements based on detected patterns.
- The results can be exported to various formats for further analysis.
- The visualization and reporting functions help users understand the results.

Remember that while technical analysis can be a valuable tool for trading, it should be used in conjunction with other forms of analysis and risk management strategies. No pattern detection system can predict market movements with 100% accuracy, and all trading involves risk.

For more information, refer to the source code in the `advanced_pattern_helpers.py` file and the examples provided in this documentation.
