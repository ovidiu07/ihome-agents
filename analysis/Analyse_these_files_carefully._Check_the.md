
# Enhanced Stock Analysis and Prediction System Prompt

## OBJECTIVE
Extend the current stock analysis system to include multi-timeframe data collection and prediction capabilities, specifically adding hourly and 15-minute data analysis alongside the existing daily analysis. Improve pattern recognition and provide predictions across all timeframes.

## IMPLEMENTATION REQUIREMENTS

### 1. Data Fetching Enhancements
Modify the `data_fetchers.py` module to:

```python
def fetch_hourly_data(symbol: str, days: int = 20) -> pd.DataFrame:
    """
    Fetch hourly OHLC data for the specified symbol for the last N days.
    
    Args:
        symbol: The stock ticker symbol
        days: Number of days to look back (default: 20)
        
    Returns:
        DataFrame with hourly OHLC data
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    df_hourly = yf.Ticker(symbol).history(
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        interval="1h"
    )
    
    df_hourly = df_hourly.reset_index()
    df_hourly["Date"] = df_hourly["Date"].dt.strftime("%Y-%m-%d %H:%M")
    
    return df_hourly

def fetch_minutes_data(symbol: str, interval: int = 15, days: int = 10) -> pd.DataFrame:
    """
    Fetch N-minute OHLC data for the specified symbol for the last M days.
    
    Args:
        symbol: The stock ticker symbol
        interval: Minute interval (default: 15)
        days: Number of days to look back (default: 10)
        
    Returns:
        DataFrame with minute-interval OHLC data
    """
    # For yfinance, valid intervals are 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h
    valid_intervals = {1: "1m", 2: "2m", 5: "5m", 15: "15m", 30: "30m", 60: "60m", 90: "90m"}
    yf_interval = valid_intervals.get(interval, "15m")
    
    # yfinance limits: 1m data available for last 7 days only, 
    # 5m, 15m, 30m for last 60 days
    max_days = 7 if interval == 1 else 60
    fetch_days = min(days, max_days)
    
    # We may need to fetch in chunks for longer periods
    end_date = datetime.now()
    start_date = end_date - timedelta(days=fetch_days)
    
    df_minutes = yf.Ticker(symbol).history(
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        interval=yf_interval
    )
    
    df_minutes = df_minutes.reset_index()
    df_minutes["Date"] = df_minutes["Date"].dt.strftime("%Y-%m-%d %H:%M")
    
    # Filter to regular trading hours
    df_minutes = df_minutes[df_minutes["Date"].str.split(" ").str[1].between("09:30", "16:00")]
    
    return df_minutes

def fetch_polygon_intraday(symbol: str, api_key: str, interval: int = 15, 
                          days: int = 10, limit: int = 1000) -> pd.DataFrame:
    """
    Alternative implementation using Polygon.io for more reliable intraday data.
    
    Args:
        symbol: The stock ticker symbol
        api_key: Polygon.io API key
        interval: Minute interval (default: 15)
        days: Number of days to look back (default: 10)
        limit: Maximum number of bars to fetch
        
    Returns:
        DataFrame with minute-interval OHLC data
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Convert to milliseconds timestamp
    start_ts = int(start_date.timestamp() * 1000)
    end_ts = int(end_date.timestamp() * 1000)
    
    url = (f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{interval}/minute/"
           f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
           f"?adjusted=true&sort=asc&limit={limit}&apiKey={api_key}")
    
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        
        if not data.get("results"):
            print(f"⚠️ No {interval}-minute data returned for the specified period.")
            return pd.DataFrame()
        
        bars = [
            {"Datetime": datetime.fromtimestamp(bar["t"] / 1000), 
             "Open": bar["o"], "High": bar["h"], "Low": bar["l"], 
             "Close": bar["c"], "Volume": bar.get("v", 0)}
            for bar in data["results"]
        ]
        
        df = pd.DataFrame(bars)
        
        # Keep only regular-hours bars
        df = df[df["Datetime"].dt.time.between(time(9, 30), time(16, 0))]
        
        # Harmonize column names
        if "Date" not in df.columns:
            df["Date"] = df["Datetime"]
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d %H:%M")
        
        return df
        
    except Exception as exc:
        print(f"❌ Error fetching {interval}-minute bars: {exc}")
        return pd.DataFrame()
```

### 2. Modify `run_analysis.py` to Include Multi-timeframe Analysis

Update the main function to fetch and analyze data at multiple timeframes:

```python
def main(symbol: str = "NVDA") -> None:
    """Run the complete pattern analysis pipeline with multi-timeframe support."""
    # ─── Environment / configuration ───────────────────────────────────────
    load_dotenv()
    poly_key: str | None = os.getenv("POLYGON_KEY")
    if not poly_key:
        print("❌  POLYGON_KEY not set in environment variables.", file=sys.stderr)
        return
    
    symbol = symbol.upper()
    lookback_daily = "3mo"  # daily history to pull
    lookback_hourly = 20    # days of hourly data
    lookback_minutes = 10   # days of 15-minute data
    mc_paths = 2_000        # Monte-Carlo paths for probabilistic forecast
    
    # ─── Fetch historical data at multiple timeframes ─────────────────────
    df_daily_hist = fetch_history(symbol, period=lookback_daily, interval="1d")
    df_hourly_hist = fetch_history(symbol, period=f"{lookback_hourly}d", interval="1h")
    
    # Fetch 15-minute data (try Polygon first, fall back to yfinance)
    df_minutes = fetch_polygon_intraday(symbol, poly_key, interval=15, days=lookback_minutes)
    if df_minutes.empty:
        print("⚠️ Falling back to yfinance for 15-minute data")
        df_minutes = fetch_minutes_data(symbol, interval=15, days=lookback_minutes)
    
    # Intraday (today)
    df_today_min = fetch_intraday_bars(symbol, poly_key, limit=150)
    
    # ─── Process daily data (existing code) ─────────────────────────────────
    # [Keep existing daily analysis code]
    
    # ─── NEW: Hourly pattern analysis ─────────────────────────────────────
    hourly_results = analyze_patterns(symbol, df_hourly_hist, df_hourly_hist.tail(48), window=12)
    
    hourly_patterns = cluster_and_keep_best(
        remove_duplicates_by_status(drop_duplicates(hourly_results["patterns"]),
            status_to_remove="Duplicate",
        ), overlap=0.7,
    )
    hourly_results["patterns"] = hourly_patterns
    
    # Refine to next-hour predictions
    hourly_results = refine_next_predictions(hourly_results, df_hourly_hist, 
                                           weight_pattern=0.5, weight_volatility=0.5)
    
    # ─── NEW: 15-minute pattern analysis ─────────────────────────────────────
    if not df_minutes.empty:
        minutes_results = analyze_patterns(symbol, df_minutes, df_minutes.tail(48), window=16)
        
        minutes_patterns = cluster_and_keep_best(
            remove_duplicates_by_status(drop_duplicates(minutes_results["patterns"]),
                status_to_remove="Duplicate",
            ), overlap=0.7,
        )
        minutes_results["patterns"] = minutes_patterns
        
        # Refine to next-interval predictions
        minutes_results = refine_next_predictions(minutes_results, df_minutes, 
                                               weight_pattern=0.6, weight_volatility=0.4)
    else:
        minutes_results = {"patterns": [], "next_prediction": None, "symbol": symbol}
    
    # ─── NEW: Probabilistic forecasts for each timeframe ─────────────────────
    # Daily forecast (existing)
    day_fcast = probabilistic_day_forecast(ohlc_df=df_combined,
        active_patterns=results["patterns"], num_mc_paths=mc_paths,
        beta_k=1.0,  # drift scaling
    )
    
    # NEW: Hourly forecast
    hour_fcast = probabilistic_timeframe_forecast(ohlc_df=df_hourly_hist,
        active_patterns=hourly_results["patterns"], num_mc_paths=mc_paths,
        beta_k=0.8,  # slightly lower drift for shorter timeframe
    )
    
    # NEW: 15-minute forecast
    if not df_minutes.empty:
        minute_fcast = probabilistic_timeframe_forecast(ohlc_df=df_minutes,
            active_patterns=minutes_results["patterns"], num_mc_paths=mc_paths,
            beta_k=0.6,  # even lower drift for shortest timeframe
        )
    else:
        minute_fcast = None
    
    # ─── Combine results into enhanced output ─────────────────────────────────
    enhanced_results = {
        "symbol": symbol,
        "patterns": results["patterns"],
        "hourly_patterns": hourly_results["patterns"],
        "minutes_patterns": minutes_results["patterns"],
        "next_prediction": results["next_prediction"],
        "hourly_prediction": hourly_results["next_prediction"],
        "minutes_prediction": minutes_results["next_prediction"],
        "stock_data": df_summary.to_dict('records'),
        "hourly_data": df_hourly_hist.tail(48).to_dict('records'),
        "minutes_data": df_minutes.tail(48).to_dict('records') if not df_minutes.empty else [],
        "probabilistic_forecasts": {
            "daily": day_fcast,
            "hourly": hour_fcast,
            "minutes": minute_fcast if minute_fcast else {"message": "No minute data available"}
        },
        "news_headlines": results.get("news_headlines", [])
    }
    
    # Export enhanced results
    export_enhanced_results(enhanced_results)
    
    # Print summary reports for each timeframe
    print("\n📊 Daily pattern summary:")
    print_summary_report(results, show_forecast=True)
    
    print("\n⏱️ Hourly pattern summary:")
    print_summary_report(hourly_results, show_forecast=True)
    
    if not df_minutes.empty:
        print("\n⏲️ 15-minute pattern summary:")
        print_summary_report(minutes_results, show_forecast=True)
```

### 3. Add New Forecasting Function for Different Timeframes

Add this to forecasting.py:

```python
def probabilistic_timeframe_forecast(ohlc_df: pd.DataFrame,
    active_patterns: List[Dict[str, Any]], num_mc_paths: int = 1000,
    atr_period: int = 14, beta_k: float = 1.0, ) -> Dict[str, Any]:
    """
    Generate a probabilistic forecast for the next period in any timeframe.
    Adapted from probabilistic_day_forecast but generalized for any timeframe.
    
    Parameters
    ----------
    ohlc_df : pd.DataFrame
        Historical bars with columns ["open","high","low","close"].
    active_patterns : list[dict]
        Result of `refine_next_predictions`; must contain fields
        {"name": str, "direction": "bullish" | "bearish"}.
    num_mc_paths : int
        How many Monte-Carlo scenarios to simulate.
    atr_period : int
        Look-back for ATR calculation.
    beta_k : float
        Scales the directional drift (higher = larger expected move).
    """
    # Implementation similar to probabilistic_day_forecast but with timeframe-specific adjustments
    # [Copy and adapt the existing implementation]
```

### 4. Enhanced Pattern Analysis

Improve the pattern detection by adding these functions to candlestick_patterns.py:

```python
def cs_engulfing(df: pd.DataFrame) -> pd.Series:
    """
    Detect bullish and bearish engulfing patterns.
    
    An engulfing pattern occurs when the current candle's body completely engulfs
    the previous candle's body.
    """
    df = _prep(df)
    
    # Calculate body ranges
    curr_body_top = df[['Open', 'Close']].max(axis=1)
    curr_body_bot = df[['Open', 'Close']].min(axis=1)
    prev_body_top = df[['Open', 'Close']].shift(1).max(axis=1)
    prev_body_bot = df[['Open', 'Close']].shift(1).min(axis=1)
    
    # Bullish engulfing: current candle is bullish and engulfs previous bearish candle
    is_bullish_engulfing = (
        (df['Close'] > df['Open']) &  # Current candle is bullish
        (df['Open'].shift(1) > df['Close'].shift(1)) &  # Previous candle is bearish
        (curr_body_bot < prev_body_bot) &  # Current body bottom below previous body bottom
        (curr_body_top > prev_body_top)  # Current body top above previous body top
    )
    
    # Bearish engulfing: current candle is bearish and engulfs previous bullish candle
    is_bearish_engulfing = (
        (df['Close'] < df['Open']) &  # Current candle is bearish
        (df['Open'].shift(1) < df['Close'].shift(1)) &  # Previous candle is bullish
        (curr_body_bot < prev_body_bot) &  # Current body bottom below previous body bottom
        (curr_body_top > prev_body_top)  # Current body top above previous body top
    )
    
    return pd.DataFrame({
        'Bullish Engulfing': is_bullish_engulfing,
        'Bearish Engulfing': is_bearish_engulfing
    })

def cs_kicker(df: pd.DataFrame) -> pd.Series:
    """
    Detect bullish and bearish kicker patterns.
    
    A kicker pattern is a powerful reversal signal where there's a gap between
    the close of one day and the open of the next in the opposite direction.
    """
    df = _prep(df)
    
    # Bullish kicker: previous bearish, current bullish with gap up
    is_bullish_kicker = (
        (df['Close'].shift(1) < df['Open'].shift(1)) &  # Previous candle is bearish
        (df['Close'] > df['Open']) &  # Current candle is bullish
        (df['Open'] > df['Close'].shift(1))  # Gap up
    )
    
    # Bearish kicker: previous bullish, current bearish with gap down
    is_bearish_kicker = (
        (df['Close'].shift(1) > df['Open'].shift(1)) &  # Previous candle is bullish
        (df['Close'] < df['Open']) &  # Current candle is bearish
        (df['Open'] < df['Close'].shift(1))  # Gap down
    )
    
    return pd.DataFrame({
        'Bullish Kicker': is_bullish_kicker,
        'Bearish Kicker': is_bearish_kicker
    })

def cs_inside_bar(df: pd.DataFrame) -> pd.Series:
    """
    Detect inside bar patterns.
    
    An inside bar is a price action pattern where the current bar's high is lower than
    the previous bar's high, and the current bar's low is higher than the previous bar's low.
    """
    df = _prep(df)
    
    is_inside_bar = (
        (df['High'] < df['High'].shift(1)) &
        (df['Low'] > df['Low'].shift(1))
    )
    
    return pd.DataFrame({
        'Inside Bar': is_inside_bar
    })

def cs_outside_bar(df: pd.DataFrame) -> pd.Series:
    """
    Detect outside bar patterns.
    
    An outside bar is a price action pattern where the current bar's high is higher than
    the previous bar's high, and the current bar's low is lower than the previous bar's low.
    """
    df = _prep(df)
    
    is_outside_bar = (
        (df['High'] > df['High'].shift(1)) &
        (df['Low'] < df['Low'].shift(1))
    )
    
    return pd.DataFrame({
        'Outside Bar': is_outside_bar
    })
```

### 5. Update the detect_candlestick_patterns Function

```python
def detect_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect various candlestick patterns in the given dataframe.
    
    Args:
        df: DataFrame with OHLC data
        
    Returns:
        DataFrame with boolean columns for each detected pattern
    """
    patterns = pd.DataFrame(index=df.index)
    
    # Single candle patterns
    patterns['Hammer'] = cs_hammer(df, confirm=True)
    patterns['Inverted Hammer'] = cs_inverted_hammer(df, confirm=True)
    patterns['Shooting Star'] = cs_shooting_star(df)
    patterns['Hanging Man'] = cs_hanging_man(df)
    patterns['Doji'] = cs_doji(df, confirm=True)
    
    # Multi-candle patterns
    patterns['Three White Soldiers'] = cs_three_white_soldiers(df)
    patterns['Three Black Crows'] = cs_three_black_crows(df)
    patterns['Morning Star'] = cs_morning_star(df)
    patterns['Evening Star'] = cs_evening_star(df)
    patterns['Bullish Harami'] = cs_bullish_harami(df)
    patterns['Bearish Harami'] = cs_bearish_harami(df)
    
    # NEW: Additional patterns
    engulfing_patterns = cs_engulfing(df)
    patterns['Bullish Engulfing'] = engulfing_patterns['Bullish Engulfing']
    patterns['Bearish Engulfing'] = engulfing_patterns['Bearish Engulfing']
    
    kicker_patterns = cs_kicker(df)
    patterns['Bullish Kicker'] = kicker_patterns['Bullish Kicker']
    patterns['Bearish Kicker'] = kicker_patterns['Bearish Kicker']
    
    inside_bar = cs_inside_bar(df)
    patterns['Inside Bar'] = inside_bar['Inside Bar']
    
    outside_bar = cs_outside_bar(df)
    patterns['Outside Bar'] = outside_bar['Outside Bar']
    
    return patterns
```

### 6. Create a New Export Function for Enhanced Results

```python
def export_enhanced_results(results: Dict[str, Any], output_dir: str = "output/model_enhanced") -> None:
    """
    Export enhanced analysis results with multi-timeframe data to a JSON file.
    
    Args:
        results: Dictionary with analysis results
        output_dir: Directory to save the output file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Format current date for filename
    today = datetime.now().strftime("%d-%m-%Y")
    symbol = results["symbol"]
    
    # Create subdirectory with date
    date_dir = os.path.join(output_dir, today)
    os.makedirs(date_dir, exist_ok=True)
    
    # Output filename
    filename = os.path.join(date_dir, f"{symbol}_Json_{today.split('-')[0]}{today.split('-')[1]}")
    
    # Write to file
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📝 Enhanced results saved to {filename}")
```

## EXPECTED OUTPUT FORMAT

The enhanced JSON output should include:

```json
{
  "symbol": "TSLA",
  "patterns": [...],  // Daily patterns (existing)
  "hourly_patterns": [...],  // NEW: Hourly patterns
  "minutes_patterns": [...],  // NEW: 15-minute patterns
  "next_prediction": {  // Daily prediction (existing)
    "direction": "bullish",
    "confidence": 0.65,
    "O": 286.52,
    "H": 300.76,
    "L": 270.38,
    "C": 287.39
  },
  "hourly_prediction": {  // NEW: Hourly prediction
    "direction": "bearish",
    "confidence": 0.58,
    "O": 286.52,
    "H": 290.25,
    "L": 283.17,
    "C": 284.92
  },
  "minutes_prediction": {  // NEW: 15-minute prediction
    "direction": "neutral",
    "confidence": 0.32,
    "O": 286.52,
    "H": 287.89,
    "L": 285.76,
    "C": 286.12
  },
  "stock_data": [...],  // Daily OHLC data (existing)
  "hourly_data": [...],  // NEW: Hourly OHLC data
  "minutes_data": [...],  // NEW: 15-minute OHLC data
  "probabilistic_forecasts": {  // NEW: Detailed probabilistic forecasts for each timeframe
    "daily": {
      "bias": "bullish",
      "prob_up": 0.6523,
      "confidence": 0.3046,
      "expected_return": 0.0152,
      "ohlc": {"o": 286.52, "h": 300.76, "l": 270.38, "c": 287.39},
      "interval_80": [275.21, 299.87]
    },
    "hourly": {
      "bias": "bearish",
      "prob_up": 0.4217,
      "confidence": 0.1566,
      "expected_return": -0.0057,
      "ohlc": {"o": 286.52, "h": 290.25, "l": 283.17, "c": 284.92},
      "interval_80": [282.35, 288.76]
    },
    "minutes": {
      "bias": "neutral",
      "prob_up": 0.5123,
      "confidence": 0.0246,
      "expected_return": 0.0008,
      "ohlc": {"o": 286.52, "h": 287.89, "l": 285.76, "c": 286.12},
      "interval_80": [285.32, 287.25]
    }
  },
  "news_headlines": [...]  // Existing news data
}
```

## IMPLEMENTATION NOTES

1. The system should maintain backward compatibility with existing code
2. For hourly data, use a 20-day lookback period
3. For 15-minute data, use a 10-day lookback period
4. Add new pattern detection functions to improve accuracy
5. Implement proper error handling for API calls
6. Ensure the output JSON format is consistent with the existing format
7. Optimize the code for performance, especially when dealing with large datasets
8. Add appropriate logging for debugging purposes

This enhanced system will provide a comprehensive multi-timeframe analysis of stock patterns and predictions, giving traders more granular insights into market movements across different time horizons.