"""Utility helpers to detect common candlestick formations.

Each function in this module accepts a pandas ``DataFrame`` containing the
standard OHLC columns (``Open``, ``High``, ``Low`` and ``Close``) and returns a
``Series`` of booleans indicating whether the pattern is present on each row.
These helpers are used by the test-suite and other pattern analysis tools.
"""

import pandas as pd
import numpy as np


def _rolling_slope(series: pd.Series, window: int = 3) -> pd.Series:
    """Return the slope of ``series`` over a moving window.

    The helper performs a simple linear regression for each rolling slice of
    ``series`` to measure short term trend direction.  The slope is used by the
    pattern detectors as a proxy for recent trend strength.
    """
    idx = np.arange(window)  # x-axis for the regression
    coefs = series.rolling(window).apply(
        lambda y: np.polyfit(idx, y, 1)[0], raw=True
    )  # 1st coef = slope
    return coefs


def _volume_confirm(df: pd.DataFrame, mult: float = 1.2) -> pd.Series:
    """Return ``True`` when volume is above-average or ``Volume`` column absent."""
    if "Volume" not in df.columns:
        return pd.Series(True, index=df.index)
    # Compare the current volume with a rolling average to weed out thin signals
    return df["Volume"] >= mult * df["Volume"].rolling(20).mean()


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    """Return a shallow copy of ``df`` to avoid side effects."""
    return df.copy()


def cs_hammer(df: pd.DataFrame, confirm: bool = False) -> pd.Series:
    """Return a boolean mask marking Hammer formations.

    Parameters
    ----------
    df : pandas.DataFrame
        OHLC dataframe.  A ``Volume`` column is optional but improves filtering.
    confirm : bool, default ``True``
        When ``True`` the pattern is only flagged if the following bar closes
        above the hammer high.

    Returns
    -------
    pandas.Series
        ``True`` where the pattern is detected.
    """
    df = _prep(df)

    # Calculate the candle anatomy ------------------------------------------------
    # Basic measurements of the candle-----------------------------------------
    real_body_top = df[["Open", "Close"]].max(axis=1)
    real_body_bot = df[["Open", "Close"]].min(axis=1)
    body_length = real_body_top - real_body_bot
    upper_shadow = df["High"] - real_body_top
    lower_shadow = real_body_bot - df["Low"]
    candle_range = df["High"] - df["Low"]

    # Core hammer criteria --------------------------------------------------------
    is_hammer = (
        (lower_shadow >= 2 * body_length)
        & (upper_shadow <= 0.6 * body_length)
        & (body_length > 0)
        & (body_length / candle_range <= 0.4)
    )
    # ── Trend & volume filters ----------------------------------------------
    # Require recent down trend and above-average volume for extra validity
    downtrend = (_rolling_slope(df["Close"], 3) < 0) if len(df) >= 3 else True
    vol_ok = _volume_confirm(df)

    mask = is_hammer & downtrend & vol_ok

    if confirm:
        # Confirmation: the next candle must close above the hammer high
        next_close_up = df["Close"].shift(-1) > df["High"]
        mask &= next_close_up

    return mask


def cs_inverted_hammer(df: pd.DataFrame, confirm: bool = False) -> pd.Series:
    """Return a mask for the *Inverted Hammer* pattern.

    The function mirrors :func:`cs_hammer` but looks for a long *upper* shadow.
    ``confirm`` behaves the same: if enabled the next candle must close higher
    than the inverted hammer's close.
    """
    df = _prep(df)

    real_body_top = df[["Open", "Close"]].max(axis=1)
    real_body_bot = df[["Open", "Close"]].min(axis=1)
    body_length = real_body_top - real_body_bot
    upper_shadow = df["High"] - real_body_top
    lower_shadow = real_body_bot - df["Low"]
    candle_range = df["High"] - df["Low"]

    # Inverted hammer criteria
    is_inverted_hammer = (
        (upper_shadow >= 2 * body_length)
        & (lower_shadow <= 0.6 * body_length)
        & (body_length > 0)
        & (body_length / candle_range <= 0.4)
    )
    downtrend = (_rolling_slope(df["Close"], 3) < 0) if len(df) >= 3 else True
    vol_ok = _volume_confirm(df)

    mask = is_inverted_hammer & downtrend & vol_ok

    if confirm:
        mask &= df["Close"].shift(-1) > df["Close"]

    # ⚠️ Bug: ``mask`` is computed but ``is_inverted_hammer`` is returned.
    # The caller never sees the volume/trend filter or confirmation logic.
    return is_inverted_hammer


def cs_shooting_star(df: pd.DataFrame) -> pd.Series:
    """Identify the bearish *Shooting Star* formation.

    The shooting star appears in up-trends and signals exhaustion.  Only the
    candle geometry is checked here – trend analysis must be done by the caller
    if desired.
    """
    df = _prep(df)

    real_body_top = df[["Open", "Close"]].max(axis=1)
    real_body_bot = df[["Open", "Close"]].min(axis=1)
    body_length = real_body_top - real_body_bot
    upper_shadow = df["High"] - real_body_top
    lower_shadow = real_body_bot - df["Low"]
    candle_range = df["High"] - df["Low"]

    # Shooting star criteria
    is_shooting_star = (
        (upper_shadow >= 2 * body_length)
        & (lower_shadow <= 0.6 * body_length)
        & (body_length > 0)
        & (df["Close"] < df["Open"])
        & (body_length / candle_range <= 0.4)
    )

    return is_shooting_star


def cs_hanging_man(df: pd.DataFrame) -> pd.Series:
    """Detect the bearish *Hanging Man* candle.

    It mirrors the :func:`cs_hammer` logic but expects the close to finish below
    the open, indicating selling pressure at the top of an up‑swing.
    """
    df = _prep(df)

    real_body_top = df[["Open", "Close"]].max(axis=1)
    real_body_bot = df[["Open", "Close"]].min(axis=1)
    body_length = real_body_top - real_body_bot
    upper_shadow = df["High"] - real_body_top
    lower_shadow = real_body_bot - df["Low"]
    candle_range = df["High"] - df["Low"]

    # Hanging man criteria
    is_hanging_man = (
        (lower_shadow >= 2 * body_length)
        & (upper_shadow <= 0.6 * body_length)
        & (body_length > 0)
        & (df["Close"] < df["Open"])
        & (body_length / candle_range <= 0.4)
    )

    return is_hanging_man


def cs_doji(df: pd.DataFrame, confirm: bool = False) -> pd.Series:
    """Detect a Doji candle.

    A doji reflects market indecision – the open and close are nearly the same.
    Optionally we can require a strong follow‑through candle in either direction
    for additional confirmation.
    """
    df = _prep(df)

    # Body size versus overall range
    body_length = abs(df["Close"] - df["Open"])
    candle_length = df["High"] - df["Low"]

    # Doji criteria
    is_doji = (body_length <= 0.1 * candle_length) & (
        candle_length > 0
    )  # Ensure there is some price movement

    vol_ok = _volume_confirm(df)
    mask = is_doji & vol_ok

    if confirm:
        # require a >-1% move the next day in either direction
        next_ret = df["Close"].shift(-1) / df["Close"] - 1
        mask &= next_ret.abs() >= 0.01

    return mask


def cs_three_white_soldiers(df: pd.DataFrame) -> pd.Series:
    """Bullish reversal composed of three long consecutive up candles."""
    df = _prep(df)
    if len(df) < 3:
        return pd.Series(False, index=df.index)

    is_bull = df["Close"] > df["Open"]
    body = (df["Close"] - df["Open"]).abs()
    rng = df["High"] - df["Low"]
    upper = df["High"] - df["Close"]

    open_within_prev1 = (
        df["Open"]
        .shift(1)
        .between(
            df[["Open", "Close"]].shift(2).min(axis=1),
            df[["Open", "Close"]].shift(2).max(axis=1),
        )
    )
    open_within_prev2 = df["Open"].between(
        df[["Open", "Close"]].shift(1).min(axis=1),
        df[["Open", "Close"]].shift(1).max(axis=1),
    )

    cond = (
        is_bull
        & is_bull.shift(1)
        & is_bull.shift(2)
        & (df["Close"] > df["Close"].shift(1))
        & (df["Close"].shift(1) > df["Close"].shift(2))
        & open_within_prev1
        & open_within_prev2
        & (upper <= body * 0.3)
        & (upper.shift(1) <= body.shift(1) * 0.3)
        & (upper.shift(2) <= body.shift(2) * 0.3)
        & (body / rng >= 0.5)
        & (body.shift(1) / rng.shift(1) >= 0.5)
        & (body.shift(2) / rng.shift(2) >= 0.5)
    )

    return cond.fillna(False)


def cs_three_black_crows(df: pd.DataFrame) -> pd.Series:
    """Bearish mirror image of :func:`cs_three_white_soldiers`."""
    df = _prep(df)

    if len(df) < 3:
        return pd.Series(False, index=df.index)

    is_bear = df["Close"] < df["Open"]
    body = (df["Close"] - df["Open"]).abs()
    rng = df["High"] - df["Low"]
    real_body_bot = df[["Open", "Close"]].min(axis=1)
    lower = real_body_bot - df["Low"]

    open_within_prev1 = (
        df["Open"]
        .shift(1)
        .between(
            df[["Open", "Close"]].shift(2).min(axis=1),
            df[["Open", "Close"]].shift(2).max(axis=1),
        )
    )
    open_within_prev2 = df["Open"].between(
        df[["Open", "Close"]].shift(1).min(axis=1),
        df[["Open", "Close"]].shift(1).max(axis=1),
    )

    cond = (
        is_bear
        & is_bear.shift(1)
        & is_bear.shift(2)
        & (df["Close"] < df["Close"].shift(1))
        & (df["Close"].shift(1) < df["Close"].shift(2))
        & open_within_prev1
        & open_within_prev2
        & (lower <= body * 0.3)
        & (lower.shift(1) <= body.shift(1) * 0.3)
        & (lower.shift(2) <= body.shift(2) * 0.3)
        & (body / rng >= 0.5)
        & (body.shift(1) / rng.shift(1) >= 0.5)
        & (body.shift(2) / rng.shift(2) >= 0.5)
    )

    return cond.fillna(False)


def cs_morning_star(df: pd.DataFrame) -> pd.Series:
    """Bullish three‑candle reversal with a gap down then strong rally."""
    if len(df) < 3:
        return pd.Series(False, index=df.index)

    body = (df["Close"] - df["Open"]).abs()
    rng = df["High"] - df["Low"]

    c1_bear = df["Close"].shift(2) < df["Open"].shift(2)
    c2_small = body.shift(1) / rng.shift(1) <= 0.3
    gap_down = df["Open"].shift(1) < df["Close"].shift(2)
    c3_bull = df["Close"] > df["Open"]
    close_into = (
        df["Close"]
        >= df["Open"].shift(2) - (df["Open"].shift(2) - df["Close"].shift(2)) / 2
    )
    gap_up = df["Open"] > df[["Open", "Close"]].shift(1).max(axis=1)

    cond = c1_bear & c2_small & gap_down & c3_bull & gap_up & close_into
    return cond.fillna(False)


def cs_evening_star(df: pd.DataFrame) -> pd.Series:
    """Bearish counterpart to :func:`cs_morning_star`."""
    if len(df) < 3:
        return pd.Series(False, index=df.index)

    body = (df["Close"] - df["Open"]).abs()
    rng = df["High"] - df["Low"]

    c1_bull = df["Close"].shift(2) > df["Open"].shift(2)
    c2_small = body.shift(1) / rng.shift(1) <= 0.3
    gap_up = df["Open"].shift(1) > df["Close"].shift(2)
    c3_bear = df["Close"] < df["Open"]
    close_into = (
        df["Close"]
        <= df["Open"].shift(2) + (df["Close"].shift(2) - df["Open"].shift(2)) / 2
    )
    gap_down = df["Open"] < df[["Open", "Close"]].shift(1).min(axis=1)

    cond = c1_bull & c2_small & gap_up & c3_bear & gap_down & close_into
    return cond.fillna(False)


def cs_bullish_harami(df: pd.DataFrame) -> pd.Series:
    """Two‑candle bullish reversal where the second body is inside the first."""
    if len(df) < 2:
        return pd.Series(False, index=df.index)

    prev_bear = df["Close"].shift(1) < df["Open"].shift(1)
    small_bull = df["Close"] > df["Open"]
    open_in = df["Open"].between(df["Close"].shift(1), df["Open"].shift(1))
    close_in = df["Close"].between(df["Close"].shift(1), df["Open"].shift(1))

    cond = prev_bear & small_bull & open_in & close_in
    return cond.fillna(False)


def cs_bearish_harami(df: pd.DataFrame) -> pd.Series:
    """Bearish variant of :func:`cs_bullish_harami`."""
    if len(df) < 2:
        return pd.Series(False, index=df.index)

    prev_bull = df["Close"].shift(1) > df["Open"].shift(1)
    small_bear = df["Close"] < df["Open"]
    open_in = df["Open"].between(df["Open"].shift(1), df["Close"].shift(1))
    close_in = df["Close"].between(df["Open"].shift(1), df["Close"].shift(1))

    cond = prev_bull & small_bear & open_in & close_in
    return cond.fillna(False)


def detect_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience wrapper that runs all pattern detectors on ``df``.

    Parameters
    ----------
    df : pandas.DataFrame
        OHLC data.  A ``Volume`` column is optional but used by some filters.

    Returns
    -------
    pandas.DataFrame
        Each column corresponds to one pattern and contains booleans.
    """
    patterns = pd.DataFrame(index=df.index)

    # Single candle patterns
    patterns["Hammer"] = cs_hammer(df, confirm=True)
    patterns["Inverted Hammer"] = cs_inverted_hammer(df, confirm=True)
    patterns["Shooting Star"] = cs_shooting_star(df)
    patterns["Hanging Man"] = cs_hanging_man(df)
    patterns["Doji"] = cs_doji(df, confirm=True)

    # Multi-candle patterns
    patterns["Three White Soldiers"] = cs_three_white_soldiers(df)
    patterns["Three Black Crows"] = cs_three_black_crows(df)
    patterns["Morning Star"] = cs_morning_star(df)
    patterns["Evening Star"] = cs_evening_star(df)
    patterns["Bullish Harami"] = cs_bullish_harami(df)
    patterns["Bearish Harami"] = cs_bearish_harami(df)

    return patterns  # table of all boolean pattern flags
