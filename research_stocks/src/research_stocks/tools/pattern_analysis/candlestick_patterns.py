# candlestick_patterns.py
# ---------------------
# Functions for detecting candlestick patterns

import pandas as pd


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare dataframe for candlestick pattern detection."""
    return df.copy()


def cs_hammer(df: pd.DataFrame) -> pd.Series:
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
    
    # Calculate body and shadow lengths
    body_length = abs(df['Close'] - df['Open'])
    upper_shadow = df['High'] - df[['Open', 'Close']].max(axis=1)
    lower_shadow = df[['Open', 'Close']].min(axis=1) - df['Low']
    
    # Hammer criteria
    is_hammer = (
        (lower_shadow > 2 * body_length) &  # Long lower shadow
        (upper_shadow < 0.1 * body_length) &  # Small or no upper shadow
        (body_length > 0)  # Ensure there is a body
    )
    
    return is_hammer


def cs_inverted_hammer(df: pd.DataFrame) -> pd.Series:
    """
    Detect inverted hammer candlestick pattern.
    
    An inverted hammer is similar to a hammer but has a long upper shadow instead.
    """
    df = _prep(df)
    
    # Calculate body and shadow lengths
    body_length = abs(df['Close'] - df['Open'])
    upper_shadow = df['High'] - df[['Open', 'Close']].max(axis=1)
    lower_shadow = df[['Open', 'Close']].min(axis=1) - df['Low']
    
    # Inverted hammer criteria
    is_inverted_hammer = (
        (upper_shadow > 2 * body_length) &  # Long upper shadow
        (lower_shadow < 0.1 * body_length) &  # Small or no lower shadow
        (body_length > 0)  # Ensure there is a body
    )
    
    return is_inverted_hammer


def cs_shooting_star(df: pd.DataFrame) -> pd.Series:
    """
    Detect shooting star candlestick pattern.
    
    A shooting star is a bearish reversal pattern that forms after an advance.
    It has a small body, a long upper shadow, and a small or nonexistent lower shadow.
    """
    df = _prep(df)
    
    # Calculate body and shadow lengths
    body_length = abs(df['Close'] - df['Open'])
    upper_shadow = df['High'] - df[['Open', 'Close']].max(axis=1)
    lower_shadow = df[['Open', 'Close']].min(axis=1) - df['Low']
    
    # Shooting star criteria
    is_shooting_star = (
        (upper_shadow > 2 * body_length) &  # Long upper shadow
        (lower_shadow < 0.1 * body_length) &  # Small or no lower shadow
        (body_length > 0) &  # Ensure there is a body
        (df['Close'] < df['Open'])  # Bearish candle
    )
    
    return is_shooting_star


def cs_hanging_man(df: pd.DataFrame) -> pd.Series:
    """
    Detect hanging man candlestick pattern.
    
    A hanging man is a bearish reversal pattern that forms after an advance.
    It has a small body, a long lower shadow, and a small or nonexistent upper shadow.
    """
    df = _prep(df)
    
    # Calculate body and shadow lengths
    body_length = abs(df['Close'] - df['Open'])
    upper_shadow = df['High'] - df[['Open', 'Close']].max(axis=1)
    lower_shadow = df[['Open', 'Close']].min(axis=1) - df['Low']
    
    # Hanging man criteria
    is_hanging_man = (
        (lower_shadow > 2 * body_length) &  # Long lower shadow
        (upper_shadow < 0.1 * body_length) &  # Small or no upper shadow
        (body_length > 0) &  # Ensure there is a body
        (df['Close'] < df['Open'])  # Bearish candle
    )
    
    return is_hanging_man


def cs_doji(df: pd.DataFrame) -> pd.Series:
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
        (body_length < 0.1 * candle_length) &  # Very small body
        (candle_length > 0)  # Ensure there is some price movement
    )
    
    return is_doji


def cs_three_white_soldiers(df: pd.DataFrame) -> pd.Series:
    """
    Detect three white soldiers candlestick pattern.
    
    Three white soldiers is a bullish reversal pattern consisting of three consecutive
    bullish candles, each closing higher than the previous.
    """
    df = _prep(df)
    result = pd.Series(False, index=df.index)
    
    # Need at least 3 candles
    if len(df) < 3:
        return result
    
    for i in range(2, len(df)):
        # Check for three consecutive bullish candles
        is_bullish_1 = df['Close'].iloc[i-2] > df['Open'].iloc[i-2]
        is_bullish_2 = df['Close'].iloc[i-1] > df['Open'].iloc[i-1]
        is_bullish_3 = df['Close'].iloc[i] > df['Open'].iloc[i]
        
        # Each candle closes higher than the previous
        higher_close_1 = df['Close'].iloc[i-1] > df['Close'].iloc[i-2]
        higher_close_2 = df['Close'].iloc[i] > df['Close'].iloc[i-1]
        
        # Pattern criteria
        if is_bullish_1 and is_bullish_2 and is_bullish_3 and higher_close_1 and higher_close_2:
            result.iloc[i] = True
    
    return result


def cs_three_black_crows(df: pd.DataFrame) -> pd.Series:
    """
    Detect three black crows candlestick pattern.
    
    Three black crows is a bearish reversal pattern consisting of three consecutive
    bearish candles, each closing lower than the previous.
    """
    df = _prep(df)
    result = pd.Series(False, index=df.index)
    
    # Need at least 3 candles
    if len(df) < 3:
        return result
    
    for i in range(2, len(df)):
        # Check for three consecutive bearish candles
        is_bearish_1 = df['Close'].iloc[i-2] < df['Open'].iloc[i-2]
        is_bearish_2 = df['Close'].iloc[i-1] < df['Open'].iloc[i-1]
        is_bearish_3 = df['Close'].iloc[i] < df['Open'].iloc[i]
        
        # Each candle closes lower than the previous
        lower_close_1 = df['Close'].iloc[i-1] < df['Close'].iloc[i-2]
        lower_close_2 = df['Close'].iloc[i] < df['Close'].iloc[i-1]
        
        # Pattern criteria
        if is_bearish_1 and is_bearish_2 and is_bearish_3 and lower_close_1 and lower_close_2:
            result.iloc[i] = True
    
    return result


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
    patterns['Hammer'] = cs_hammer(df)
    patterns['Inverted Hammer'] = cs_inverted_hammer(df)
    patterns['Shooting Star'] = cs_shooting_star(df)
    patterns['Hanging Man'] = cs_hanging_man(df)
    patterns['Doji'] = cs_doji(df)
    
    # Multi-candle patterns
    patterns['Three White Soldiers'] = cs_three_white_soldiers(df)
    patterns['Three Black Crows'] = cs_three_black_crows(df)
    
    return patterns