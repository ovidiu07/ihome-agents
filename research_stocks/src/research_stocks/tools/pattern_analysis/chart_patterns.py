# chart_patterns.py
# ---------------
# Functions for detecting chart patterns

import pandas as pd
import numpy as np

from .utils import slope


def detect_pivots(df: pd.DataFrame, left: int = 3, right: int = 3, 
                 min_diff: float = 0.005, tolerate_equal: bool = True) -> pd.DataFrame:
    """
    Detect pivot points (highs and lows) in price data.

    Args:
        df: DataFrame with OHLC data
        left: Number of bars to look left
        right: Number of bars to look right
        min_diff: Minimum price difference to consider as a pivot
        tolerate_equal: Whether to tolerate equal values when finding pivots

    Returns:
        DataFrame with pivot information
    """
    highs = df['High'].values
    lows = df['Low'].values

    # Initialize arrays for pivot highs and lows
    pivot_highs = np.zeros(len(highs))
    pivot_lows = np.zeros(len(lows))

    # Detect pivot highs
    for i in range(left, len(highs) - right):
        if tolerate_equal:
            is_pivot_high = all(highs[i] >= highs[i-j] for j in range(1, left+1)) and \
                           all(highs[i] >= highs[i+j] for j in range(1, right+1))
        else:
            is_pivot_high = all(highs[i] > highs[i-j] for j in range(1, left+1)) and \
                           all(highs[i] > highs[i+j] for j in range(1, right+1))

        if is_pivot_high and (i == 0 or abs(highs[i] - highs[np.where(pivot_highs[:i] > 0)[0][-1] if any(pivot_highs[:i] > 0) else 0]) > min_diff * highs[i]):
            pivot_highs[i] = highs[i]

    # Detect pivot lows
    for i in range(left, len(lows) - right):
        if tolerate_equal:
            is_pivot_low = all(lows[i] <= lows[i-j] for j in range(1, left+1)) and \
                          all(lows[i] <= lows[i+j] for j in range(1, right+1))
        else:
            is_pivot_low = all(lows[i] < lows[i-j] for j in range(1, left+1)) and \
                          all(lows[i] < lows[i+j] for j in range(1, right+1))

        if is_pivot_low and (i == 0 or abs(lows[i] - lows[np.where(pivot_lows[:i] > 0)[0][-1] if any(pivot_lows[:i] > 0) else 0]) > min_diff * lows[i]):
            pivot_lows[i] = lows[i]

    # Create result DataFrame
    pivots = pd.DataFrame({
        'Date': df['Date'],
        'High': df['High'],
        'Low': df['Low'],
        'Pivot_High': pivot_highs,
        'Pivot_Low': pivot_lows
    })

    return pivots


def detect_head_shoulders_pivot(df: pd.DataFrame, pivots: pd.DataFrame) -> list:
    """
    Detect head and shoulders patterns using pivot points.

    Args:
        df: DataFrame with OHLC data
        pivots: DataFrame with pivot information

    Returns:
        List of detected patterns
    """
    patterns = []

    # Get pivot highs and their indices
    pivot_high_idx = pivots.index[pivots['Pivot_High'] > 0].tolist()

    # Need at least 5 pivot highs to form a head and shoulders pattern
    if len(pivot_high_idx) < 5:
        return patterns

    # Check each possible head and shoulders pattern
    for i in range(len(pivot_high_idx) - 4):
        # Get the 5 consecutive pivot highs
        idx1, idx2, idx3, idx4, idx5 = pivot_high_idx[i:i+5]

        # Get the pivot high values
        h1 = pivots.loc[idx1, 'Pivot_High']
        h2 = pivots.loc[idx2, 'Pivot_High']
        h3 = pivots.loc[idx3, 'Pivot_High']
        h4 = pivots.loc[idx4, 'Pivot_High']
        h5 = pivots.loc[idx5, 'Pivot_High']

        # Head and shoulders criteria:
        # 1. h3 > h1 and h3 > h5 (head is higher than shoulders)
        # 2. h1 ≈ h5 (shoulders are approximately equal height)
        # 3. h2 < h1 and h4 < h5 (troughs between head and shoulders)

        # Check if it's a head and shoulders pattern
        if (h3 > h1 and h3 > h5 and 
            0.9 <= h1/h5 <= 1.1 and  # Shoulders within 10% of each other
            h2 < h1 and h4 < h5):

            # Calculate neckline
            neckline = min(h1, h5)

            # Calculate pattern height
            height = h3 - neckline

            # Add pattern to results
            pattern = {
                'pattern': 'Head and Shoulders',
                'start_date': df.loc[idx1, 'Date'],
                'end_date': df.loc[idx5, 'Date'],
                'height': height,
                'direction': 'bearish',
                'value': height / neckline * 100,  # Pattern significance as percentage
                'status': 'Confirmed'
            }
            patterns.append(pattern)

    return patterns


def detect_double_tops_bottoms_pivot(df: pd.DataFrame, pivots: pd.DataFrame) -> list:
    """
    Detect double top and double bottom patterns using pivot points.

    Args:
        df: DataFrame with OHLC data
        pivots: DataFrame with pivot information

    Returns:
        List of detected patterns
    """
    patterns = []

    # Get pivot highs and lows and their indices
    pivot_high_idx = pivots.index[pivots['Pivot_High'] > 0].tolist()
    pivot_low_idx = pivots.index[pivots['Pivot_Low'] > 0].tolist()

    # Need at least 2 pivot highs/lows to form a double top/bottom
    if len(pivot_high_idx) < 2 or len(pivot_low_idx) < 2:
        return patterns

    # Check for double tops
    for i in range(len(pivot_high_idx) - 1):
        idx1, idx2 = pivot_high_idx[i], pivot_high_idx[i+1]

        # Skip if the pivots are too close
        if idx2 - idx1 < 5:
            continue

        h1 = pivots.loc[idx1, 'Pivot_High']
        h2 = pivots.loc[idx2, 'Pivot_High']

        # Double top criteria:
        # 1. h1 ≈ h2 (tops are approximately equal height)
        # 2. There should be a significant trough between the tops

        # Check if it's a double top
        if 0.95 <= h1/h2 <= 1.05:  # Tops within 5% of each other
            # Find the lowest point between the two tops
            between_idx = df.loc[idx1:idx2].index
            lowest_between = df.loc[between_idx, 'Low'].min()

            # Calculate pattern height
            height = h1 - lowest_between

            # Add pattern to results if height is significant
            if height > 0.02 * h1:  # Height is at least 2% of price
                pattern = {
                    'pattern': 'Double Top',
                    'start_date': df.loc[idx1, 'Date'],
                    'end_date': df.loc[idx2, 'Date'],
                    'height': height,
                    'direction': 'bearish',
                    'value': height / h1 * 100,  # Pattern significance as percentage
                    'status': 'Confirmed'
                }
                patterns.append(pattern)

    # Check for double bottoms
    for i in range(len(pivot_low_idx) - 1):
        idx1, idx2 = pivot_low_idx[i], pivot_low_idx[i+1]

        # Skip if the pivots are too close
        if idx2 - idx1 < 5:
            continue

        l1 = pivots.loc[idx1, 'Pivot_Low']
        l2 = pivots.loc[idx2, 'Pivot_Low']

        # Double bottom criteria:
        # 1. l1 ≈ l2 (bottoms are approximately equal)
        # 2. There should be a significant peak between the bottoms

        # Check if it's a double bottom
        if 0.95 <= l1/l2 <= 1.05:  # Bottoms within 5% of each other
            # Find the highest point between the two bottoms
            between_idx = df.loc[idx1:idx2].index
            highest_between = df.loc[between_idx, 'High'].max()

            # Calculate pattern height
            height = highest_between - l1

            # Add pattern to results if height is significant
            if height > 0.02 * l1:  # Height is at least 2% of price
                pattern = {
                    'pattern': 'Double Bottom',
                    'start_date': df.loc[idx1, 'Date'],
                    'end_date': df.loc[idx2, 'Date'],
                    'height': height,
                    'direction': 'bullish',
                    'value': height / l1 * 100,  # Pattern significance as percentage
                    'status': 'Confirmed'
                }
                patterns.append(pattern)

    return patterns


def detect_triangle_pivot(df: pd.DataFrame, pivots: pd.DataFrame, window: int = 10) -> list:
    """
    Detect triangle patterns (ascending, descending, symmetrical) using pivot points.

    Args:
        df: DataFrame with OHLC data
        pivots: DataFrame with pivot information
        window: Window size for pattern detection

    Returns:
        List of detected patterns
    """
    patterns = []

    # Get pivot highs and lows and their indices
    pivot_high_idx = pivots.index[pivots['Pivot_High'] > 0].tolist()
    pivot_low_idx = pivots.index[pivots['Pivot_Low'] > 0].tolist()

    # Need at least 3 pivot highs/lows to form a triangle
    if len(pivot_high_idx) < 3 or len(pivot_low_idx) < 3:
        return patterns

    # Check each window of data
    for i in range(len(df) - window):
        window_df = df.iloc[i:i+window]
        window_pivots = pivots.iloc[i:i+window]

        # Get pivot highs and lows in this window
        window_high_idx = window_pivots.index[window_pivots['Pivot_High'] > 0].tolist()
        window_low_idx = window_pivots.index[window_pivots['Pivot_Low'] > 0].tolist()

        # Need at least 3 pivot highs/lows in the window
        if len(window_high_idx) < 3 or len(window_low_idx) < 3:
            continue

        # Get the pivot high and low values
        highs = window_pivots.loc[window_high_idx, 'Pivot_High'].values
        lows = window_pivots.loc[window_low_idx, 'Pivot_Low'].values

        # Calculate the slopes of the trend lines
        high_slope = slope(pd.Series(highs))
        low_slope = slope(pd.Series(lows))

        # Triangle criteria based on slopes
        if high_slope < -0.001 and low_slope > 0.001:
            # Converging trend lines with descending highs and ascending lows
            pattern_type = 'Symmetrical Triangle'
            direction = 'neutral'
        elif abs(high_slope) < 0.001 and low_slope > 0.001:
            # Horizontal resistance and ascending support
            pattern_type = 'Ascending Triangle'
            direction = 'bullish'
        elif high_slope < -0.001 and abs(low_slope) < 0.001:
            # Descending resistance and horizontal support
            pattern_type = 'Descending Triangle'
            direction = 'bearish'
        else:
            # Not a triangle pattern
            continue

        # Calculate pattern height (difference between first high and low)
        height = highs[0] - lows[0]

        # Add pattern to results
        pattern = {
            'pattern': pattern_type,
            'start_date': window_df.iloc[0]['Date'],
            'end_date': window_df.iloc[-1]['Date'],
            'height': height,
            'direction': direction,
            'value': height / lows[0] * 100,  # Pattern significance as percentage
            'status': 'Confirmed'
        }
        patterns.append(pattern)

    return patterns
