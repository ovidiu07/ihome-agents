"""Experimental helpers for pattern based price forecasting."""

import functools
import math
import os
import logging
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from .utils import get_pattern_reliability

logger = logging.getLogger(__name__)
if os.getenv("FORECAST_DEBUG") == "1":
    logging.basicConfig(level=logging.DEBUG)


# forecasting.py
# -------------
# Functions for forecasting based on pattern analysis


def _breakout_confirmed(df: pd.DataFrame, pattern: Dict) -> bool:
    """Return ``True`` if the bars following ``pattern`` confirm a breakout."""
    # Find the end date of the pattern in the dataframe
    try:
        end_idx = df[df["Date"] == pattern["end_date"]].index[0]
    except (IndexError, KeyError):
        return False

    # Skip if we're at the end of the dataframe
    if end_idx >= len(df) - 1:
        return False

    # Get the next few bars after pattern completion
    next_bars = df.iloc[end_idx + 1 : min(end_idx + 6, len(df))]

    if next_bars.empty:
        return False

    # Check for breakout confirmation based on pattern direction
    if pattern["direction"] == "bullish":
        # For bullish patterns, check if price moves above the pattern high
        pattern_high = df.loc[df["Date"] == pattern["end_date"], "High"].iloc[0]
        return any(next_bars["High"] > pattern_high * 1.01)  # 1% above pattern high

    elif pattern["direction"] == "bearish":
        # For bearish patterns, check if price moves below the pattern low
        pattern_low = df.loc[df["Date"] == pattern["end_date"], "Low"].iloc[0]
        return any(next_bars["Low"] < pattern_low * 0.99)  # 1% below pattern low

    return False


def _label_patterns(df: pd.DataFrame, patterns: List[Dict]) -> pd.DataFrame:
    """Return ``df`` with additional columns describing active patterns."""
    # Create a copy of the dataframe
    labeled_df = df.copy()

    # Add pattern columns
    labeled_df["Pattern"] = ""
    labeled_df["Pattern_Direction"] = ""
    labeled_df["Pattern_Score"] = 0.0

    # Label each pattern
    for pattern in patterns:
        try:
            start_idx = labeled_df[labeled_df["Date"] == pattern["start_date"]].index[0]
            end_idx = labeled_df[labeled_df["Date"] == pattern["end_date"]].index[0]

            # Label all rows in the pattern range
            for idx in range(start_idx, end_idx + 1):
                if labeled_df.loc[idx, "Pattern"] == "":
                    labeled_df.loc[idx, "Pattern"] = pattern["pattern"]
                    labeled_df.loc[idx, "Pattern_Direction"] = pattern["direction"]
                    labeled_df.loc[idx, "Pattern_Score"] = pattern.get("value", 0)
                else:
                    # If there's already a pattern, append the new one
                    labeled_df.loc[idx, "Pattern"] += f", {pattern['pattern']}"
                    labeled_df.loc[
                        idx, "Pattern_Direction"
                    ] += f", {pattern['direction']}"
                    labeled_df.loc[idx, "Pattern_Score"] = max(
                        labeled_df.loc[idx, "Pattern_Score"], pattern.get("value", 0)
                    )
        except (IndexError, KeyError):
            continue

    return labeled_df


def _decay(t_delta_h: float, half_life_h: float) -> float:
    """Exponential decay factor used for time weighting."""
    return math.exp(-t_delta_h / half_life_h)


# ---------------------------------------------------------------------------
# Improved high/close logic — empirical excursions conditioned on bias
# ---------------------------------------------------------------------------
def refine_next_predictions(
    results: Dict[str, Any],
    df: pd.DataFrame,
    *,
    weight_pattern: float = 0.4,
    weight_volatility: float = 0.6,
    atr_window: int = 14,
    half_life_h: float = 8.0,
    hi_lo_q_hi: float = 0.75,
    hi_lo_q_lo: float = 0.25,
) -> Dict[str, Any]:
    """
    One-bar OHLC forecast.
    Open & Low logic unchanged; High and Close now derive from
    *conditional* empirical excursions (↑ for bull bias, ↓ for bear bias).
    """
    if abs(weight_pattern + weight_volatility - 1.0) > 1e-6:
        raise ValueError("weights must sum to 1")

    out = results.copy()
    if not results.get("patterns") or len(df) < atr_window + 5:
        return out

    latest = df.iloc[-1]
    close = latest["Close"]
    now_ts = pd.to_datetime(latest["Date"])

    # ── 1. EW-ATR & series-level metrics ──────────────────────────────
    tr = np.maximum(
        df["High"] - df["Low"],
        np.maximum(
            (df["High"] - df["Close"].shift()).abs(),
            (df["Low"] - df["Close"].shift()).abs(),
        ),
    )
    atr_series = tr.ewm(alpha=1 / atr_window, adjust=False).mean()
    atr = atr_series.iloc[-1]

    # intraday excursion ratios
    up_ex = (df["High"] - df[["Open", "Close"]].max(axis=1)) / atr_series
    dn_ex = (df[["Open", "Close"]].min(axis=1) - df["Low"]) / atr_series
    up_move = (df["Close"] - df["Open"]) / atr_series  # + on up-days
    dn_move = (df["Open"] - df["Close"]) / atr_series  # + on down-days

    pos_mask = up_move >= 0
    neg_mask = dn_move >= 0

    # ── 2. Pattern-derived probability (unchanged) ───────────────────
    reliab_map = get_pattern_reliability()
    log_odds = 0.0
    for p in results["patterns"]:
        age_h = max(
            (now_ts - pd.to_datetime(p["end_date"])).total_seconds() / 3600, 0.0
        )
        decay = math.exp(-age_h / half_life_h)
        base_p = max(0.01, min(0.99, reliab_map.get(p["pattern"], 0.55)))
        strength = p.get("value", 50) / 100
        contrib = math.log(base_p / (1 - base_p)) * strength * decay
        log_odds += contrib if p["direction"] == "bullish" else -contrib

    prob_up = 1 / (1 + math.exp(-log_odds))
    direction = (
        "bullish" if prob_up > 0.55 else "bearish" if prob_up < 0.45 else "neutral"
    )
    conf = abs(prob_up - 0.5) * 2  # 0-1

    # choose conditional quantiles based on confidence
    def _q(base: float, span: float = 0.3) -> float:
        return np.clip(base + span * conf, 0.05, 0.95)

    # ── 3. Price targets (new High/Close logic) ───────────────────────
    if direction == "bullish":
        q_hi = _q(0.55)
        q_close = _q(0.50, 0.25)
        hi_exc = np.nanquantile(up_ex[pos_mask], q_hi) if pos_mask.any() else hi_lo_q_hi
        cl_mv = np.nanquantile(up_move[pos_mask], q_close) if pos_mask.any() else 0.5
        high = close + hi_exc * atr * weight_volatility
        low = (
            close - np.nanquantile(dn_ex, hi_lo_q_lo) * atr * 0.5
        )  # unchanged low logic
        close_next = close + cl_mv * atr * weight_pattern

    elif direction == "bearish":
        q_hi = 1 - _q(0.55)  # smaller upside
        q_close = _q(0.50, 0.25)
        hi_exc = np.nanquantile(up_ex[neg_mask], q_hi) if neg_mask.any() else hi_lo_q_lo
        cl_mv = np.nanquantile(dn_move[neg_mask], q_close) if neg_mask.any() else 0.5
        low = (
            close - np.nanquantile(dn_ex[neg_mask], _q(0.55)) * atr * weight_volatility
        )
        high = close + hi_exc * atr * 0.4  # limited upside
        close_next = close - cl_mv * atr * weight_pattern

    else:  # sideways
        hi_exc = np.nanquantile(up_ex, 0.5)
        dn_exc = np.nanquantile(dn_ex, 0.5)
        high = close + hi_exc * atr * 0.5
        low = close - dn_exc * atr * 0.5
        close_next = close

    # ── 4. Gap / open (unchanged) ─────────────────────────────────────
    gaps = ((df["Open"] / df["Close"].shift()) - 1).dropna()
    gap_pct = (
        np.random.choice(gaps.values) if not gaps.empty else np.random.normal(0, 0.002)
    )
    open_next = close * (1 + gap_pct)

    # ensure high ≥ close & open  ; low ≤ ...
    high = max(high, open_next, close_next)
    low = min(low, open_next, close_next)

    out["next_prediction"] = {
        "direction": direction,
        "confidence": round(conf, 2),
        "prob_up": round(prob_up, 3),
        "O": round(float(open_next), 2),
        "H": round(float(high), 2),
        "L": round(float(low), 2),
        "C": round(float(close_next), 2),
    }
    return out


class _OHLCForecaster:
    """
    Internal class for OHLC forecasting using statistical methods.
    """

    def __init__(self, opens, highs, lows, closes):
        """
        Initialize the forecaster with historical OHLC data.

        Args:
            opens: Series of open prices
            highs: Series of high prices
            lows: Series of low prices
            closes: Series of close prices
        """
        self.opens = opens
        self.highs = highs
        self.lows = lows
        self.closes = closes

        # Calculate returns and ranges
        self.close_returns = closes.pct_change().dropna()
        self.high_low_ranges = (highs - lows) / closes
        self.open_close_ranges = abs(opens - closes) / closes

        # Calculate statistics
        self.mean_return = self.close_returns.mean()
        self.std_return = self.close_returns.std()
        self.mean_hl_range = self.high_low_ranges.mean()
        self.mean_oc_range = self.open_close_ranges.mean()

        # Calculate correlations
        self.corr_matrix = pd.DataFrame(
            {"open": opens, "high": highs, "low": lows, "close": closes}
        ).corr()

    def predict(self, days=1):
        """
        Generate OHLC predictions for the specified number of days.

        Args:
            days: Number of days to forecast

        Returns:
            DataFrame with predicted OHLC values
        """
        predictions = []
        last_close = self.closes.iloc[-1]

        for _ in range(days):
            # Predict close using random return from normal distribution
            close_return = np.random.normal(self.mean_return, self.std_return)
            pred_close = last_close * (1 + close_return)

            # Predict high-low range
            hl_range = np.random.normal(self.mean_hl_range, self.high_low_ranges.std())
            hl_range = max(0.005, hl_range)  # Ensure positive range

            # Predict open-close range
            oc_range = np.random.normal(
                self.mean_oc_range, self.open_close_ranges.std()
            )
            oc_range = max(0.001, oc_range)  # Ensure positive range

            # Determine if open is above or below close
            if np.random.random() > 0.5:
                # Bullish day (close > open)
                pred_open = pred_close / (1 + oc_range)

                # High is above both open and close
                pred_high = pred_close * (1 + hl_range / 2)

                # Low is below open
                pred_low = pred_open * (1 - hl_range / 2)
            else:
                # Bearish day (open > close)
                pred_open = pred_close * (1 + oc_range)

                # High is above open
                pred_high = pred_open * (1 + hl_range / 2)

                # Low is below close
                pred_low = pred_close * (1 - hl_range / 2)

            # Ensure high >= max(open, close) and low <= min(open, close)
            pred_high = max(pred_high, pred_open, pred_close)
            pred_low = min(pred_low, pred_open, pred_close)

            # Add prediction
            predictions.append(
                {"O": pred_open, "H": pred_high, "L": pred_low, "C": pred_close}
            )

            # Update last close for next iteration
            last_close = pred_close

        return predictions[0] if days == 1 else predictions


def build_feature_stack(
    df_hist: pd.DataFrame,
    df_today_min: Optional[pd.DataFrame],
    daily_patterns: List[Dict],
    intraday_patterns: List[Dict],
    vwap_trend: str,
    morning_cutoff: str = "10:30",
) -> Dict[str, Any]:
    """
    Build a feature stack for forecasting.

    Args:
        df_hist: DataFrame with historical daily OHLC data
        df_today_min: DataFrame with today's intraday OHLC data (can be None)
        daily_patterns: List of daily pattern dictionaries
        intraday_patterns: List of intraday pattern dictionaries
        vwap_trend: VWAP trend direction ("UP" or "DOWN")
        morning_cutoff: Time cutoff for morning session

    Returns:
        Dictionary with features for forecasting
    """
    features = {}

    # Helper function to calculate bias score from patterns
    def _bias_score(patts: List[Dict]) -> float:
        if not patts:
            return 0.0

        bullish = sum(1 for p in patts if p["direction"] == "bullish")
        bearish = sum(1 for p in patts if p["direction"] == "bearish")

        if bullish == bearish:
            return 0.0

        # Calculate normalized score between -1 and 1
        total = bullish + bearish
        return (bullish - bearish) / total

    # Historical features
    if not df_hist.empty:
        # Price momentum
        features["price_momentum"] = df_hist["Close"].pct_change(5).iloc[-1]

        # Volatility
        features["volatility"] = (
            df_hist["High"]
            .sub(df_hist["Low"])
            .div(df_hist["Close"])
            .rolling(10, min_periods=1)
            .mean()
            .iloc[-1]
        )

        # Daily pattern bias
        features["daily_pattern_bias"] = _bias_score(daily_patterns)

        # Recent performance
        features["week_return"] = df_hist["Close"].pct_change(5).iloc[-1]
        features["month_return"] = df_hist["Close"].pct_change(20).iloc[-1]

    # Intraday features
    if df_today_min is not None and not df_today_min.empty:
        # Morning vs. full day performance
        morning_data = df_today_min[
            df_today_min["Date"].str.contains(morning_cutoff, regex=False)
        ]

        if not morning_data.empty:
            morning_open = morning_data.iloc[0]["Open"]
            morning_close = morning_data.iloc[-1]["Close"]
            features["morning_return"] = (morning_close / morning_open) - 1

        # Intraday pattern bias
        features["intraday_pattern_bias"] = _bias_score(intraday_patterns)

        # Intraday volatility
        features["intraday_volatility"] = (
            df_today_min["High"].max() / df_today_min["Low"].min() - 1
        )

        # VWAP trend
        features["vwap_trend"] = 1 if vwap_trend == "UP" else -1

    # Combined features
    features["combined_bias"] = (
        features.get("daily_pattern_bias", 0) * 0.6
        + features.get("intraday_pattern_bias", 0) * 0.4
    )

    return features


@functools.lru_cache(maxsize=1)
def _load_pattern_stats() -> pd.DataFrame:
    """
    Builds a lookup dataframe:
        Multi-Index (pattern_name, direction)  ->  column 'p'  (success probability)
    The base reliability numbers come from utils.get_pattern_reliability().
    For a bearish version we assume symmetry: p_bear = 1 - p_bull.
    Laplace smoothing is applied (Beta(1,1) prior) to avoid 0/1 extremes.
    """
    reliab: Dict[str, float] = get_pattern_reliability()
    records: List[Tuple[str, str, float]] = []
    for pat, r in reliab.items():
        r = max(0.01, min(0.99, r))  # clamp
        # +1 / (n+2)  with n=1  gives the same clamp, but keep explicit
        records.append((pat, "bullish", r))
        records.append((pat, "bearish", 1.0 - r))
    df = pd.DataFrame(records, columns=["pattern", "direction", "p"])
    df.set_index(["pattern", "direction"], inplace=True)
    return df


def probabilistic_day_forecast(
    ohlc_df: pd.DataFrame,
    active_patterns: List[Dict[str, Any]],
    num_mc_paths: int = 1000,
    atr_period: int = 14,
    beta_k: float = 1.0,
) -> Dict[str, Any]:
    """
    Generate a probabilistic forecast for the next trading day.

    Parameters
    ----------
    ohlc_df : pd.DataFrame
        Historical daily bars with columns ["open","high","low","close"].
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
    # ------------------------------------------------------------------
    # NEW: normalise incoming price data so later code can rely on it.
    # ------------------------------------------------------------------
    ohlc_df = _normalize_ohlc(ohlc_df)
    active_patterns = _normalize_pattern_df(active_patterns)

    stats = _load_pattern_stats()
    last_close = ohlc_df["close"].iloc[-1]

    # ---- 1. Combine pattern odds (Bayesian sum of log-odds) -------------
    if active_patterns.empty:
        prob_up = 0.5
    else:
        log_odds_sum = 0.0
        # Dampening factor to prevent extreme probabilities
        dampening_factor = 0.7
        pattern_count = len(active_patterns)

        for _, p in active_patterns.iterrows():
            key = (p["name"], p["direction"])
            p_prob = stats["p"].get(key, 0.55)  # default mild edge
            p_prob = max(0.01, min(0.99, p_prob))
            sign = +1 if p["direction"] == "bullish" else -1
            # Apply dampening factor to each log-odds contribution
            log_odds_sum += (
                math.log(p_prob / (1 - p_prob))
                * sign
                * dampening_factor
                / max(1, math.sqrt(pattern_count))
            )
        prob_up = 1 / (1 + math.exp(-log_odds_sum))

    # Cap confidence at 90% to acknowledge inherent market uncertainty
    confidence = min(0.9, abs(prob_up - 0.5) * 2.0)  # 0.0 … 0.9
    bias = "bullish" if prob_up > 0.55 else "bearish" if prob_up < 0.45 else "neutral"

    # ---- 2. Historical ATR and expected drift ---------------------------
    tr = np.maximum(
        ohlc_df["high"] - ohlc_df["low"],
        np.maximum(
            (ohlc_df["high"] - ohlc_df["close"].shift()).abs(),
            (ohlc_df["low"] - ohlc_df["close"].shift()).abs(),
        ),
    )
    atr = tr.rolling(atr_period, min_periods=1).mean().iloc[-1]
    atr_pct = atr / last_close if last_close > 0 else 0.0

    mu = (prob_up - 0.5) * 2.0 * beta_k * atr_pct  # signed drift

    # ---- 3. Monte-Carlo simulation of next close ------------------------
    returns = np.log(ohlc_df["close"]).diff().dropna()
    if returns.empty or returns.std(ddof=0) == 0.0:
        # fall-back: thin normal noise
        returns = pd.Series(np.random.normal(0, 1e-4, size=50))

    sampled_cc = np.random.choice(returns, size=num_mc_paths, replace=True)
    sampled_cc = np.exp(sampled_cc + mu) - 1.0  # shift by drift

    close_samples = last_close * (1.0 + sampled_cc)
    # directional intraday excursion proportional to ATR
    high_samples = (
        np.maximum(last_close, close_samples)
        + np.random.uniform(0.1, 0.5, num_mc_paths) * atr
    )
    low_samples = (
        np.minimum(last_close, close_samples)
        - np.random.uniform(0.1, 0.5, num_mc_paths) * atr
    )

    # ---- 4. Point estimates & interval ----------------------------------
    open_ = last_close
    close_ = float(np.median(close_samples))
    high_ = float(np.quantile(high_samples, 0.75))
    low_ = float(np.quantile(low_samples, 0.25))
    p10, p90 = np.quantile(close_samples, [0.10, 0.90])

    return {
        "bias": bias,
        "prob_up": round(float(prob_up), 4),
        "confidence": round(float(confidence), 4),
        "expected_return": round(float(mu), 4),
        "ohlc": {"o": open_, "h": high_, "l": low_, "c": close_},
        "interval_80": (round(float(p10), 4), round(float(p90), 4)),
        "patterns": (
            active_patterns["name"].tolist() if not active_patterns.empty else []
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────
def _normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the OHLC DataFrame has lowercase column names and always contains a
    `close` column (falling back to any available adjusted-close variant).
    """
    # 1. lower-case all column names
    df = df.rename(columns={c: c.lower() for c in df.columns})

    # 2. guarantee presence of "close"
    if "close" not in df.columns:
        for alt in ("adj close", "adjclose", "adjusted_close"):
            if alt in df.columns:
                df["close"] = df[alt]
                break
        else:
            raise KeyError(
                "'close' column not found in OHLC data (even after normalising)"
            )

    return df


def _normalize_pattern_df(df: Any) -> pd.DataFrame:
    """
    Return a DataFrame that always contains (at least) the columns
    `name` and `direction`, both in lowercase.

    Accepted inputs:
        • None
        • pandas.DataFrame
        • list / tuple / set of dicts
        • list / tuple of (name, direction) pairs
        • plain dict of column → list
    Any other type raises TypeError.

    Direction aliases understood: dir, trend, side
    Name     aliases understood: pattern, type
    """
    import pandas as pd  # local import keeps the signature usable in typing-only contexts

    # ------------------------------------------------------------------ #
    # 1.  Bring *any* input into a DataFrame or an empty fallback.
    # ------------------------------------------------------------------ #
    if df is None:  # explicit “no patterns”
        return pd.DataFrame(columns=["name", "direction"])

    if isinstance(df, pd.DataFrame):  # already OK → shallow copy
        work = df.copy()

    elif isinstance(df, (list, tuple, set)):
        if not df:  # empty sequence
            return pd.DataFrame(columns=["name", "direction"])

        first = next(iter(df))

        # (a) sequence of (name, direction) pairs  → build directly
        if isinstance(first, (list, tuple)) and len(first) >= 2:
            work = pd.DataFrame(df, columns=["name", "direction"])

        # (b) sequence of mapping-like objects      → DataFrame(list(..))
        else:
            work = pd.DataFrame(list(df))

    elif isinstance(df, dict):  # raw dict of columns
        work = pd.DataFrame(df)

    else:
        raise TypeError(f"Unsupported patterns container type: {type(df).__name__}")

    # ------------------------------------------------------------------ #
    # 2.  Column canonisation.
    # ------------------------------------------------------------------ #
    work.columns = [str(c).lower() for c in work.columns]

    if "name" not in work.columns:
        for alt in ("pattern", "type"):
            if alt in work.columns:
                work["name"] = work[alt]
                break

    if "direction" not in work.columns:
        for alt in ("dir", "trend", "side"):
            if alt in work.columns:
                work["direction"] = work[alt]
                break

    # supply defaults / guardrails
    if "direction" not in work.columns:
        work["direction"] = "unknown"

    if "name" not in work.columns:
        raise KeyError(
            "Pattern DataFrame lacks required column 'name' after normalisation"
        )

    # standardise textual content
    work["direction"] = (
        work["direction"].astype(str, copy=False).str.lower().str.strip()
    )

    # keep original extra columns (if any) after the canonical two
    ordered_cols = ["name", "direction"] + [
        c for c in work.columns if c not in ("name", "direction")
    ]
    return work[ordered_cols]


def _df_from_finnhub_candles(candles_block: Dict[str, Any]) -> pd.DataFrame:
    """Convert Finnhub candle payload into a tidy DataFrame."""

    required = ["o", "h", "l", "c", "v", "t"]
    for key in required:
        if key not in candles_block:
            raise ValueError(f"candles block missing key: {key}")

    lengths = {len(candles_block[key]) for key in required}
    if len(lengths) != 1:
        raise ValueError("candle vectors must be of equal length")

    df = pd.DataFrame(
        {
            "Date": candles_block["t"],
            "Open": candles_block["o"],
            "High": candles_block["h"],
            "Low": candles_block["l"],
            "Close": candles_block["c"],
            "Volume": candles_block["v"],
        }
    )

    tvals = candles_block["t"]
    if tvals and isinstance(tvals[0], (int, float, np.integer, np.floating)):
        dt = pd.to_datetime(tvals, unit="s", utc=True)
    else:
        dt = pd.to_datetime(tvals, utc=True)
    df["Date"] = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return df


def _patterns_from_finnhub(raw_patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Simplify Finnhub pattern objects."""

    if not raw_patterns:
        return []

    out: List[Dict[str, Any]] = []
    for p in raw_patterns:
        pat = p.get("patternname")
        direction = str(p.get("patterntype", "")).lower()
        value = float(p.get("value", 1.0))
        start_ts = p.get("atime") or p.get("start_time")
        end_ts = p.get("dtime") or p.get("end_time")

        start_dt = pd.to_datetime(start_ts, unit="s", utc=True).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        end_dt = pd.to_datetime(end_ts, unit="s", utc=True).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

        out.append(
            {
                "pattern": pat,
                "direction": direction,
                "value": value,
                "start_date": start_dt,
                "end_date": end_dt,
            }
        )

    return out


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Return Relative Strength Index (RSI)."""
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    gain = up.ewm(alpha=1 / period, adjust=False).mean()
    loss = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Return MACD, signal line, and histogram."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist


def next_prediction_from_finnhub(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return a single OHLC forecast by stacking multiple timeframes.

    The payload dictionary should contain 1-minute, 1-hour, 1-day and
    1-week candle data under the keys ``fintech_minutes``, ``fintech_hourly``,
    ``fintech_daily`` and ``fintech_weekly`` respectively. Each block follows
    the structure returned by :func:`fetch_all`, namely ``{"candles": {"o", "h",
    "l", "c"}}``. The function computes a small forecast for each timeframe
    using ATR driven rules and blends them weighted by confidence.
    """

    def _extract(data: Dict[str, Any]) -> Optional[Tuple[List[float], List[float], List[float]]]:
        if not data or "candles" not in data:
            return None
        cdl = data["candles"]
        return cdl.get("h", []), cdl.get("l", []), cdl.get("c", [])

    def _atr(high: List[float], low: List[float], close: List[float], window: int = 14) -> float:
        if len(close) < 2:
            return 0.0
        prev_close = np.concatenate([[close[0]], close[:-1]])
        tr = np.maximum(np.array(high) - np.array(low),
                        np.maximum(np.abs(np.array(high) - prev_close),
                                   np.abs(np.array(low) - prev_close)))
        if len(tr) < window:
            return float(np.mean(tr))
        return float(np.mean(tr[-window:]))

    def _mini_forecast(high: List[float], low: List[float], close: List[float]) -> Dict[str, float]:
        n = len(close)
        if n < 2:
            return {}
        atr = _atr(high, low, close)
        last_close = close[-1]
        ref_close = close[-10] if n > 10 else close[0]
        delta = last_close - ref_close

        if atr > 0 and delta > atr:
            trend = "UP"
        elif atr > 0 and delta < -atr:
            trend = "DOWN"
        else:
            trend = "SIDEWAYS"

        if atr == 0:
            atr = last_close * 0.001

        if trend == "UP":
            close_f = last_close + 0.5 * atr
            high_f = last_close + atr
            low_f = last_close - 0.2 * atr
        elif trend == "DOWN":
            close_f = last_close - 0.5 * atr
            high_f = last_close + 0.2 * atr
            low_f = last_close - atr
        else:
            close_f = last_close
            high_f = last_close + 0.3 * atr
            low_f = last_close - 0.3 * atr

        conf = 0.0 if atr == 0 else min(1.0, abs(delta) / atr)
        if trend == "SIDEWAYS":
            conf *= 0.2

        return {
            "open": last_close,
            "high": high_f,
            "low": low_f,
            "close": close_f,
            "confidence": float(round(conf, 3)),
            "trend": trend,
        }

    tf_data = {
        "1min": _extract(payload.get("fintech_minutes")),
        "1h": _extract(payload.get("fintech_hourly")),
        "1d": _extract(payload.get("fintech_daily")),
        "1w": _extract(payload.get("fintech_weekly")),
    }

    forecasts: Dict[str, Dict[str, float]] = {}
    for tf, series in tf_data.items():
        if series is None:
            continue
        h, l, c = series
        if len(c) < 2:
            continue
        forecasts[tf] = _mini_forecast(h, l, c)

    if not forecasts:
        raise ValueError("No candle data available for forecasting")

    total_weight = sum(f["confidence"] for f in forecasts.values())
    if total_weight <= 0:
        total_weight = float(len(forecasts))

    def _wavg(key: str) -> float:
        return sum(f[key] * f["confidence"] for f in forecasts.values()) / total_weight

    final_open = _wavg("open")
    final_close = _wavg("close")
    final_high = _wavg("high")
    final_low = _wavg("low")

    final_high = max(final_high, final_open, final_close)
    final_low = min(final_low, final_open, final_close)

    return {
        "timeframe": "multi",
        "open": round(final_open, 2),
        "high": round(final_high, 2),
        "low": round(final_low, 2),
        "close": round(final_close, 2),
        "confidence_scores": {tf: f["confidence"] for tf, f in forecasts.items()},
        "component_forecasts": forecasts,
    }


if __name__ == "__main__":
    # Basic self-test
    from pprint import pprint

    SAMPLE_PAYLOAD = {
        "fintech_daily": {
            "symbol": "AAPL",
            "resolution": "D",
            "last_updated_utc": "2025-07-12T10:47:04.505889Z",
            "candles": {
                "o": [
                    204.39,
                    200.6,
                    203.5,
                    199.08,
                    199.73,
                    197.3,
                    197.2,
                    195.94,
                    198.235,
                    201.625,
                    202.59,
                    201.45,
                    201.43,
                    201.89,
                    202.01,
                    206.665,
                    208.91,
                    212.145,
                    212.68,
                    210.1,
                    209.53,
                    210.505
                ],
                "h": [
                    206.0,
                    204.35,
                    204.5,
                    199.68,
                    200.37,
                    198.685,
                    198.39,
                    197.57,
                    201.7,
                    202.3,
                    203.44,
                    203.67,
                    202.64,
                    203.22,
                    207.39,
                    210.1865,
                    213.34,
                    214.65,
                    216.23,
                    211.43,
                    211.33,
                    213.48
                ],
                "l": [
                    200.02,
                    200.57,
                    198.41,
                    197.3601,
                    195.7,
                    196.5636,
                    195.21,
                    195.07,
                    196.855,
                    198.96,
                    200.2,
                    200.6201,
                    199.46,
                    200.0,
                    199.2607,
                    206.1401,
                    208.14,
                    211.8101,
                    208.8,
                    208.45,
                    207.22,
                    210.03
                ],
                "c": [
                    201.45,
                    202.67,
                    198.78,
                    199.2,
                    196.45,
                    198.42,
                    195.64,
                    196.58,
                    201.0,
                    201.5,
                    200.3,
                    201.56,
                    201.0,
                    201.08,
                    205.17,
                    207.82,
                    212.44,
                    213.55,
                    209.95,
                    210.01,
                    211.14,
                    212.41
                ],
                "v": [
                    72862557.0,
                    54672608.0,
                    60989857.0,
                    43904635.0,
                    51447349.0,
                    43020691.0,
                    38856152.0,
                    45394689.0,
                    96813542.0,
                    55814272.0,
                    54064033.0,
                    39525730.0,
                    50799121.0,
                    73188571.0,
                    91912816.0,
                    78788867.0,
                    67941811.0,
                    34955836.0,
                    50228984.0,
                    42848928.0,
                    48749367.0,
                    44443635.0
                ],
                "t": [
                    "2025-06-09T00:00:00Z",
                    "2025-06-10T00:00:00Z",
                    "2025-06-11T00:00:00Z",
                    "2025-06-12T00:00:00Z",
                    "2025-06-13T00:00:00Z",
                    "2025-06-16T00:00:00Z",
                    "2025-06-17T00:00:00Z",
                    "2025-06-18T00:00:00Z",
                    "2025-06-20T00:00:00Z",
                    "2025-06-23T00:00:00Z",
                    "2025-06-24T00:00:00Z",
                    "2025-06-25T00:00:00Z",
                    "2025-06-26T00:00:00Z",
                    "2025-06-27T00:00:00Z",
                    "2025-06-30T00:00:00Z",
                    "2025-07-01T00:00:00Z",
                    "2025-07-02T00:00:00Z",
                    "2025-07-03T00:00:00Z",
                    "2025-07-07T00:00:00Z",
                    "2025-07-08T00:00:00Z",
                    "2025-07-09T00:00:00Z",
                    "2025-07-10T00:00:00Z"
                ],
                "s": "ok"
            },
            "patterns": [
                {
                    "aprice": 193.25,
                    "atime": 1746576000,
                    "bprice": 213.94,
                    "btime": 1747180800,
                    "cprice": 193.46,
                    "ctime": 1747958400,
                    "dprice": 206.24,
                    "dtime": 1748995200,
                    "entry": 205.14,
                    "entry_date": 1749168000,
                    "intersect_price": 193.97281938325995,
                    "intersect_time": 1751932800,
                    "mature": 1,
                    "patternname": "Triangle",
                    "patterntype": "bullish",
                    "profit1": 225.83,
                    "profit2": 0,
                    "sortTime": 1748995200,
                    "status": "complete",
                    "stoploss": 193.0567,
                    "symbol": "AAPL.US",
                    "terminal": 0
                },
                {
                    "aprice": 214.56,
                    "atime": 1746057600,
                    "bprice": 193.25,
                    "btime": 1746576000,
                    "cprice": 213.94,
                    "ctime": 1747180800,
                    "dprice": 193.46,
                    "dtime": 1747958400,
                    "end_price": 193.5475,
                    "end_time": 1748822400,
                    "entry": 193.5475,
                    "eprice": 206.24,
                    "etime": 1748995200,
                    "mature": 0,
                    "patternname": "Triple Top",
                    "patterntype": "bearish",
                    "profit1": 170.404,
                    "profit2": 0,
                    "sortTime": 1748822400,
                    "start_price": 193.0925,
                    "start_time": 1745452800,
                    "status": "incomplete",
                    "stoploss": 216.691,
                    "symbol": "AAPL.US",
                    "terminal": 0
                },
                {
                    "aprice": 213.58,
                    "atime": 1745971200,
                    "dprice": 197.02,
                    "dtime": 1746489600,
                    "mature": 1,
                    "patternname": "two black gapping",
                    "patterntype": "bearish",
                    "sortTime": 1746489600,
                    "status": "complete",
                    "symbol": "AAPL.US"
                },
                {
                    "aprice": 219.38,
                    "atime": 1737417600,
                    "bprice": 247.19,
                    "btime": 1738281600,
                    "cprice": 225.7,
                    "ctime": 1738540800,
                    "dprice": 0,
                    "dtime": 0,
                    "end_price": 247.19,
                    "end_time": 1740096000,
                    "entry": 247.19,
                    "eprice": 0,
                    "etime": 0,
                    "mature": 0,
                    "patternname": "Double Bottom",
                    "patterntype": "bullish",
                    "profit1": 275,
                    "profit2": 0,
                    "sortTime": 1740096000,
                    "start_price": 247.19,
                    "start_time": 1736121600,
                    "status": "failed",
                    "stoploss": 216.599,
                    "symbol": "AAPL.US",
                    "terminal": 0
                }
            ],
            "support_resistance": {
                "levels": [
                    169.21009826660156,
                    199.2606964111328,
                    216.22999572753906,
                    230.1999969482422,
                    260.1000061035156
                ]
            },
            "aggregate_indicator": {
                "trend": {
                    "adx": 17.760926651100156,
                    "trending": False
                },
                "technicalAnalysis": {
                    "count": {
                        "buy": 5,
                        "neutral": 8,
                        "sell": 3
                    },
                    "signal": "neutral"
                }
            },
            "technical_indicators": {
                "c": [
                    201.45,
                    202.67,
                    198.78,
                    199.2,
                    196.45,
                    198.42,
                    195.64,
                    196.58,
                    201,
                    201.5,
                    200.3,
                    201.56,
                    201,
                    201.08,
                    205.17,
                    207.82,
                    212.44,
                    213.55,
                    209.95,
                    210.01,
                    211.14,
                    212.41
                ],
                "h": [
                    206,
                    204.35,
                    204.5,
                    199.68,
                    200.37,
                    198.685,
                    198.39,
                    197.57,
                    201.7,
                    202.3,
                    203.44,
                    203.67,
                    202.64,
                    203.22,
                    207.39,
                    210.1865,
                    213.34,
                    214.65,
                    216.23,
                    211.43,
                    211.33,
                    213.48
                ],
                "l": [
                    200.02,
                    200.57,
                    198.41,
                    197.3601,
                    195.7,
                    196.5636,
                    195.21,
                    195.07,
                    196.855,
                    198.96,
                    200.2,
                    200.6201,
                    199.46,
                    200,
                    199.2607,
                    206.1401,
                    208.14,
                    211.8101,
                    208.8,
                    208.45,
                    207.22,
                    210.03
                ],
                "o": [
                    204.39,
                    200.6,
                    203.5,
                    199.08,
                    199.73,
                    197.3,
                    197.2,
                    195.94,
                    198.235,
                    201.625,
                    202.59,
                    201.45,
                    201.43,
                    201.89,
                    202.01,
                    206.665,
                    208.91,
                    212.145,
                    212.68,
                    210.1,
                    209.53,
                    210.505
                ],
                "rsi": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0
                ],
                "s": "ok",
                "t": [
                    1749427200,
                    1749513600,
                    1749600000,
                    1749686400,
                    1749772800,
                    1750032000,
                    1750118400,
                    1750204800,
                    1750377600,
                    1750636800,
                    1750723200,
                    1750809600,
                    1750896000,
                    1750982400,
                    1751241600,
                    1751328000,
                    1751414400,
                    1751500800,
                    1751846400,
                    1751932800,
                    1752019200,
                    1752105600,
                    1752192000
                ],
                "v": [
                    72862557,
                    54672608,
                    60989857,
                    43904635,
                    51447349,
                    43020691,
                    38856152,
                    45394689,
                    96813542,
                    55814272,
                    54064033,
                    39525730,
                    50799121,
                    73188571,
                    91912816,
                    78788867,
                    67941811,
                    34955836,
                    50228984,
                    42848928,
                    48749367,
                    44443635,
                    39765812
                ]
            }
        },
        "fintech_hourly": {
            "symbol": "AAPL",
            "resolution": "60",
            "last_updated_utc": "2025-07-12T10:47:10.942557Z",
            "candles": {
                "o": [
                    210.65,
                    210.3,
                    209.7,
                    209.95,
                    210.2,
                    210.0,
                    211.16,
                    212.53,
                    212.29,
                    212.06,
                    212.4,
                    212.94,
                    212.33,
                    212.21,
                    212.29,
                    212.32,
                    210.811,
                    210.961
                ],
                "h": [
                    210.65,
                    210.4,
                    210.0,
                    210.26,
                    210.26,
                    211.28,
                    213.48,
                    213.315,
                    212.395,
                    213.16,
                    213.14,
                    213.45,
                    212.44,
                    212.35,
                    212.33,
                    212.5,
                    211.001,
                    211.121
                ],
                "l": [
                    210.1,
                    209.62,
                    209.7,
                    209.92,
                    209.81,
                    209.85,
                    210.71,
                    212.23,
                    211.84,
                    211.93,
                    212.315,
                    212.05,
                    212.11,
                    212.12,
                    212.29,
                    212.3,
                    210.309,
                    210.339
                ],
                "c": [
                    210.24,
                    209.72,
                    210.0,
                    210.12,
                    209.99,
                    211.275,
                    212.36,
                    212.39,
                    212.01,
                    212.39,
                    212.935,
                    212.42,
                    212.22,
                    212.22,
                    212.31,
                    212.5,
                    210.941,
                    210.421
                ],
                "v": [
                    37851.0,
                    47288.0,
                    13242.0,
                    30701.0,
                    144902.0,
                    5025437.0,
                    9487417.0,
                    5353170.0,
                    3929444.0,
                    3263078.0,
                    2739833.0,
                    6354044.0,
                    577556.0,
                    72958.0,
                    18082.0,
                    14803.0,
                    50947.0,
                    37766.0
                ],
                "t": [
                    "2025-07-10T08:00:00Z",
                    "2025-07-10T09:00:00Z",
                    "2025-07-10T10:00:00Z",
                    "2025-07-10T11:00:00Z",
                    "2025-07-10T12:00:00Z",
                    "2025-07-10T13:00:00Z",
                    "2025-07-10T14:00:00Z",
                    "2025-07-10T15:00:00Z",
                    "2025-07-10T16:00:00Z",
                    "2025-07-10T17:00:00Z",
                    "2025-07-10T18:00:00Z",
                    "2025-07-10T19:00:00Z",
                    "2025-07-10T20:00:00Z",
                    "2025-07-10T21:00:00Z",
                    "2025-07-10T22:00:00Z",
                    "2025-07-10T23:00:00Z",
                    "2025-07-11T08:00:00Z",
                    "2025-07-11T09:00:00Z"
                ],
                "s": "ok"
            },
            "patterns": [
                {
                    "aprice": 209.849,
                    "atime": 1752238800,
                    "bprice": 212.141,
                    "btime": 1752242400,
                    "cprice": 210.389,
                    "ctime": 1752264000,
                    "dprice": 0,
                    "dtime": 0,
                    "end_price": 212.141,
                    "end_time": 1752274800,
                    "entry": 212.141,
                    "eprice": 0,
                    "etime": 0,
                    "mature": 0,
                    "patternname": "Double Bottom",
                    "patterntype": "bullish",
                    "profit1": 214.433,
                    "profit2": 0,
                    "sortTime": 1752274800,
                    "start_price": 212.141,
                    "start_time": 1752188400,
                    "status": "incomplete",
                    "stoploss": 209.6198,
                    "symbol": "AAPL.US",
                    "terminal": 0
                },
                {
                    "aprice": 212.5,
                    "atime": 1752188400,
                    "bprice": 210.289,
                    "btime": 1752228000,
                    "cprice": 212.141,
                    "ctime": 1752242400,
                    "dprice": 210.389,
                    "dtime": 1752264000,
                    "entry": 0,
                    "entry_date": 1751443200,
                    "intersect_price": 210.62466433566425,
                    "intersect_time": 1752343200,
                    "mature": 1,
                    "patternname": "Triangle",
                    "patterntype": "unknown",
                    "profit1": 0,
                    "profit2": 0,
                    "sortTime": 1752264000,
                    "status": "incomplete",
                    "stoploss": 0,
                    "symbol": "AAPL.US",
                    "terminal": 0
                }
            ],
            "support_resistance": {
                "levels": [
                    207.22000122070312,
                    209.85000610351562,
                    211.67100524902344,
                    213.6199951171875,
                    216.22999572753906
                ]
            },
            "aggregate_indicator": {
                "trend": {
                    "adx": 11.668724288397813,
                    "trending": False
                },
                "technicalAnalysis": {
                    "count": {
                        "buy": 2,
                        "neutral": 8,
                        "sell": 5
                    },
                    "signal": "neutral"
                }
            },
            "technical_indicators": {
                "c": [
                    210.24,
                    209.72,
                    210,
                    210.12,
                    209.99,
                    211.275,
                    212.36,
                    212.39,
                    212.01,
                    212.39,
                    212.935,
                    212.42,
                    212.22,
                    212.22,
                    212.31,
                    212.5,
                    210.941,
                    210.421
                ],
                "h": [
                    210.65,
                    210.4,
                    210,
                    210.26,
                    210.26,
                    211.28,
                    213.48,
                    213.315,
                    212.395,
                    213.16,
                    213.14,
                    213.45,
                    212.44,
                    212.35,
                    212.33,
                    212.5,
                    211.001,
                    211.121
                ],
                "l": [
                    210.1,
                    209.62,
                    209.7,
                    209.92,
                    209.81,
                    209.85,
                    210.71,
                    212.23,
                    211.84,
                    211.93,
                    212.315,
                    212.05,
                    212.11,
                    212.12,
                    212.29,
                    212.3,
                    210.309,
                    210.339
                ],
                "o": [
                    210.65,
                    210.3,
                    209.7,
                    209.95,
                    210.2,
                    210,
                    211.16,
                    212.53,
                    212.29,
                    212.06,
                    212.4,
                    212.94,
                    212.33,
                    212.21,
                    212.29,
                    212.32,
                    210.811,
                    210.961
                ],
                "rsi": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0
                ],
                "s": "ok",
                "t": [
                    1752134400,
                    1752138000,
                    1752141600,
                    1752145200,
                    1752148800,
                    1752152400,
                    1752156000,
                    1752159600,
                    1752163200,
                    1752166800,
                    1752170400,
                    1752174000,
                    1752177600,
                    1752181200,
                    1752184800,
                    1752188400,
                    1752220800,
                    1752224400
                ],
                "v": [
                    37851,
                    47288,
                    13242,
                    30701,
                    144902,
                    5025437,
                    9487417,
                    5353170,
                    3929444,
                    3263078,
                    2739833,
                    6354044,
                    577556,
                    72958,
                    18082,
                    14803,
                    50947,
                    37766
                ]
            }
        },
    }

    #df_test = _df_from_finnhub_candles(SAMPLE_PAYLOAD["fintech_daily"]["candles"])

    pred = next_prediction_from_finnhub(SAMPLE_PAYLOAD)

    pprint(pred)
