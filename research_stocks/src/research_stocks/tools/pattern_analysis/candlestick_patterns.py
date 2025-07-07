# candlestick_patterns.py
# ---------------------
# Functions for detecting candlestick patterns

import pandas as pd
import numpy as np


def _rolling_slope(series: pd.Series, window: int = 3) -> pd.Series:
  """Simple linear-regression slope over a rolling window."""
  idx = np.arange(window)
  coefs = (
    series.rolling(window)
    .apply(lambda y: np.polyfit(idx, y, 1)[0], raw=True)
  )
  return coefs

def _volume_confirm(df: pd.DataFrame, mult: float = 1.2) -> pd.Series:
  """True when today’s volume beats `mult` × 20-day average."""
  return df["Volume"] >= mult * df["Volume"].rolling(20).mean()


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` to avoid mutating the caller."""
    return df.copy()


def cs_hammer(df: pd.DataFrame,  confirm: bool = True) -> pd.Series:
    """
    Detect hammer candlestick pattern.
    
    A hammer is a single-candle bullish reversal pattern that forms after a decline.
    It has a small body, a long lower shadow, and a small or nonexistent upper shadow.
    
    Args:
        df: DataFrame with OHLC data
        
    Returns:
        Series with boolean values indicating pattern presence
    """
    df = _prep(df)
    
    real_body_top = df[['Open', 'Close']].max(axis=1)
    real_body_bot = df[['Open', 'Close']].min(axis=1)
    body_length = real_body_top - real_body_bot
    upper_shadow = df['High'] - real_body_top
    lower_shadow = real_body_bot - df['Low']
    candle_range = df['High'] - df['Low']
    
    # Hammer criteria
    is_hammer = (
        (lower_shadow >= 2 * body_length) &
        (upper_shadow <= 0.6 * body_length) &
        (body_length > 0) &
        (body_length / candle_range <= 0.4)
    )
    # ── NEW trend & volume filters ────────────────────────────────────
    downtrend = _rolling_slope(df['Close'], 3) < 0
    vol_ok = _volume_confirm(df)

    mask = is_hammer & downtrend & vol_ok

    if confirm:
        # next day closes above hammer high
        next_close_up = df['Close'].shift(-1) > df['High']
        mask &= next_close_up

    return mask


def cs_inverted_hammer(df: pd.DataFrame,  confirm: bool = True) -> pd.Series:
    """
    Detect inverted hammer candlestick pattern.
    
    An inverted hammer is similar to a hammer but has a long upper shadow instead.
    """
    df = _prep(df)
    
    real_body_top = df[['Open', 'Close']].max(axis=1)
    real_body_bot = df[['Open', 'Close']].min(axis=1)
    body_length = real_body_top - real_body_bot
    upper_shadow = df['High'] - real_body_top
    lower_shadow = real_body_bot - df['Low']
    candle_range = df['High'] - df['Low']
    
    # Inverted hammer criteria
    is_inverted_hammer = (
        (upper_shadow >= 2 * body_length) &
        (lower_shadow <= 0.6 * body_length) &
        (body_length > 0) &
        (body_length / candle_range <= 0.4)
    )
    downtrend = _rolling_slope(df['Close'], 3) < 0
    vol_ok = _volume_confirm(df)

    mask = is_inverted_hammer & downtrend & vol_ok

    if confirm:
        mask &= df['Close'].shift(-1) > df['Close']

    return is_inverted_hammer


def cs_shooting_star(df: pd.DataFrame) -> pd.Series:
    """
    Detect shooting star candlestick pattern.
    
    A shooting star is a bearish reversal pattern that forms after an advance.
    It has a small body, a long upper shadow, and a small or nonexistent lower shadow.
    """
    df = _prep(df)
    
    real_body_top = df[['Open', 'Close']].max(axis=1)
    real_body_bot = df[['Open', 'Close']].min(axis=1)
    body_length = real_body_top - real_body_bot
    upper_shadow = df['High'] - real_body_top
    lower_shadow = real_body_bot - df['Low']
    candle_range = df['High'] - df['Low']
    
    # Shooting star criteria
    is_shooting_star = (
        (upper_shadow >= 2 * body_length) &
        (lower_shadow <= 0.6 * body_length) &
        (body_length > 0) &
        (df['Close'] < df['Open']) &
        (body_length / candle_range <= 0.4)
    )
    
    return is_shooting_star


def cs_hanging_man(df: pd.DataFrame) -> pd.Series:
    """
    Detect hanging man candlestick pattern.
    
    A hanging man is a bearish reversal pattern that forms after an advance.
    It has a small body, a long lower shadow, and a small or nonexistent upper shadow.
    """
    df = _prep(df)
    
    real_body_top = df[['Open', 'Close']].max(axis=1)
    real_body_bot = df[['Open', 'Close']].min(axis=1)
    body_length = real_body_top - real_body_bot
    upper_shadow = df['High'] - real_body_top
    lower_shadow = real_body_bot - df['Low']
    candle_range = df['High'] - df['Low']
    
    # Hanging man criteria
    is_hanging_man = (
        (lower_shadow >= 2 * body_length) &
        (upper_shadow <= 0.6 * body_length) &
        (body_length > 0) &
        (df['Close'] < df['Open']) &
        (body_length / candle_range <= 0.4)
    )
    
    return is_hanging_man


def cs_doji(df: pd.DataFrame, confirm: bool = True) -> pd.Series:
    """
    Detect doji candlestick pattern.
    
    A doji has a very small body, indicating indecision in the market.
    The open and close prices are very close or equal.
    """
    df = _prep(df)
    
    # Calculate body and total candle lengths
    body_length = abs(df['Close'] - df['Open'])
    candle_length = df['High'] - df['Low']
    
    # Doji criteria
    is_doji = (
        (body_length <= 0.1 * candle_length) &
        (candle_length > 0)  # Ensure there is some price movement
    )

    vol_ok = _volume_confirm(df)
    mask = is_doji & vol_ok

    if confirm:
        # require a >-1% move the next day in either direction
        next_ret = df['Close'].shift(-1) / df['Close'] - 1
        mask &= next_ret.abs() >= 0.01

    return mask


def cs_three_white_soldiers(df: pd.DataFrame) -> pd.Series:
    """
    Detect three white soldiers candlestick pattern.
    
    Three white soldiers is a bullish reversal pattern consisting of three consecutive
    bullish candles, each closing higher than the previous.
    """
    df = _prep(df)
    if len(df) < 3:
        return pd.Series(False, index=df.index)

    is_bull = df['Close'] > df['Open']
    body = (df['Close'] - df['Open']).abs()
    rng = df['High'] - df['Low']
    upper = df['High'] - df['Close']

    open_within_prev1 = df['Open'].shift(1).between(
        df[['Open', 'Close']].shift(2).min(axis=1),
        df[['Open', 'Close']].shift(2).max(axis=1))
    open_within_prev2 = df['Open'].between(
        df[['Open', 'Close']].shift(1).min(axis=1),
        df[['Open', 'Close']].shift(1).max(axis=1))

    cond = (
        is_bull & is_bull.shift(1) & is_bull.shift(2) &
        (df['Close'] > df['Close'].shift(1)) &
        (df['Close'].shift(1) > df['Close'].shift(2)) &
        open_within_prev1 & open_within_prev2 &
        (upper <= body * 0.3) &
        (upper.shift(1) <= body.shift(1) * 0.3) &
        (upper.shift(2) <= body.shift(2) * 0.3) &
        (body / rng >= 0.5) &
        (body.shift(1) / rng.shift(1) >= 0.5) &
        (body.shift(2) / rng.shift(2) >= 0.5)
    )

    return cond.fillna(False)


def cs_three_black_crows(df: pd.DataFrame) -> pd.Series:
    """
    Detect three black crows candlestick pattern.
    
    Three black crows is a bearish reversal pattern consisting of three consecutive
    bearish candles, each closing lower than the previous.
    """
    df = _prep(df)

    if len(df) < 3:
        return pd.Series(False, index=df.index)

    is_bear = df['Close'] < df['Open']
    body = (df['Close'] - df['Open']).abs()
    rng = df['High'] - df['Low']
    real_body_bot = df[['Open', 'Close']].min(axis=1)
    lower = real_body_bot - df['Low']

    open_within_prev1 = df['Open'].shift(1).between(
        df[['Open', 'Close']].shift(2).min(axis=1),
        df[['Open', 'Close']].shift(2).max(axis=1))
    open_within_prev2 = df['Open'].between(
        df[['Open', 'Close']].shift(1).min(axis=1),
        df[['Open', 'Close']].shift(1).max(axis=1))

    cond = (
        is_bear & is_bear.shift(1) & is_bear.shift(2) &
        (df['Close'] < df['Close'].shift(1)) &
        (df['Close'].shift(1) < df['Close'].shift(2)) &
        open_within_prev1 & open_within_prev2 &
        (lower <= body * 0.3) &
        (lower.shift(1) <= body.shift(1) * 0.3) &
        (lower.shift(2) <= body.shift(2) * 0.3) &
        (body / rng >= 0.5) &
        (body.shift(1) / rng.shift(1) >= 0.5) &
        (body.shift(2) / rng.shift(2) >= 0.5)
    )

    return cond.fillna(False)


def cs_morning_star(df: pd.DataFrame) -> pd.Series:
    """Detect Morning Star pattern (3 candles)."""
    if len(df) < 3:
        return pd.Series(False, index=df.index)

    body = (df['Close'] - df['Open']).abs()
    rng = df['High'] - df['Low']

    c1_bear = df['Close'].shift(2) < df['Open'].shift(2)
    c2_small = body.shift(1) / rng.shift(1) <= 0.3
    gap_down = df['Open'].shift(1) < df['Close'].shift(2)
    c3_bull = df['Close'] > df['Open']
    close_into = df['Close'] >= df['Open'].shift(2) - (df['Open'].shift(2) - df['Close'].shift(2)) / 2
    gap_up = df['Open'] > df[['Open', 'Close']].shift(1).max(axis=1)

    cond = c1_bear & c2_small & gap_down & c3_bull & gap_up & close_into
    return cond.fillna(False)


def cs_evening_star(df: pd.DataFrame) -> pd.Series:
    """Detect Evening Star pattern (3 candles)."""
    if len(df) < 3:
        return pd.Series(False, index=df.index)

    body = (df['Close'] - df['Open']).abs()
    rng = df['High'] - df['Low']

    c1_bull = df['Close'].shift(2) > df['Open'].shift(2)
    c2_small = body.shift(1) / rng.shift(1) <= 0.3
    gap_up = df['Open'].shift(1) > df['Close'].shift(2)
    c3_bear = df['Close'] < df['Open']
    close_into = df['Close'] <= df['Open'].shift(2) + (df['Close'].shift(2) - df['Open'].shift(2)) / 2
    gap_down = df['Open'] < df[['Open', 'Close']].shift(1).min(axis=1)

    cond = c1_bull & c2_small & gap_up & c3_bear & gap_down & close_into
    return cond.fillna(False)


def cs_bullish_harami(df: pd.DataFrame) -> pd.Series:
    """Detect Bullish Harami pattern (2 candles)."""
    if len(df) < 2:
        return pd.Series(False, index=df.index)

    prev_bear = df['Close'].shift(1) < df['Open'].shift(1)
    small_bull = (df['Close'] > df['Open'])
    open_in = df['Open'].between(df['Close'].shift(1), df['Open'].shift(1))
    close_in = df['Close'].between(df['Close'].shift(1), df['Open'].shift(1))

    cond = prev_bear & small_bull & open_in & close_in
    return cond.fillna(False)


def cs_bearish_harami(df: pd.DataFrame) -> pd.Series:
    """Detect Bearish Harami pattern (2 candles)."""
    if len(df) < 2:
        return pd.Series(False, index=df.index)

    prev_bull = df['Close'].shift(1) > df['Open'].shift(1)
    small_bear = df['Close'] < df['Open']
    open_in = df['Open'].between(df['Open'].shift(1), df['Close'].shift(1))
    close_in = df['Close'].between(df['Open'].shift(1), df['Close'].shift(1))

    cond = prev_bull & small_bear & open_in & close_in
    return cond.fillna(False)


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
    patterns['Hammer'] = cs_hammer(df, confirm= True)
    patterns['Inverted Hammer'] = cs_inverted_hammer(df, confirm= True)
    patterns['Shooting Star'] = cs_shooting_star(df)
    patterns['Hanging Man'] = cs_hanging_man(df)
    patterns['Doji'] = cs_doji(df, confirm= True)

    # Multi-candle patterns
    patterns['Three White Soldiers'] = cs_three_white_soldiers(df)
    patterns['Three Black Crows'] = cs_three_black_crows(df)
    patterns['Morning Star'] = cs_morning_star(df)
    patterns['Evening Star'] = cs_evening_star(df)
    patterns['Bullish Harami'] = cs_bullish_harami(df)
    patterns['Bearish Harami'] = cs_bearish_harami(df)
    
    return patterns