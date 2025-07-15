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


def probability_blender(probs: List[float], weights: Optional[List[float]] = None) -> float:
    """Return weighted average of probabilities."""
    if not probs:
        return 0.5
    if weights is None:
        return float(sum(probs) / len(probs))
    if len(weights) != len(probs):
        raise ValueError("weights length mismatch")
    total = sum(weights)
    if total == 0:
        return float(sum(probs) / len(probs))
    return float(sum(p * w for p, w in zip(probs, weights)) / total)


def build_feature_dict(df: pd.DataFrame) -> Dict[str, float]:
    """Return a small feature dictionary derived from ``df``."""
    df = _normalize_ohlc(df)
    close = df["close"]
    high = df["high"]
    low = df["low"]

    tr = np.maximum(high - low, np.maximum((high - close.shift()).abs(), (low - close.shift()).abs()))
    atr14 = tr.rolling(14, min_periods=1).mean().iloc[-1]
    rsi14 = _rsi(close, 14).iloc[-1]
    macd, signal, hist = _macd(close)

    return {
        "atr14": round(float(atr14), 4),
        "rsi14": round(float(rsi14), 2),
        "macd": round(float(macd.iloc[-1]), 4),
        "macd_signal": round(float(signal.iloc[-1]), 4),
        "macd_hist": round(float(hist.iloc[-1]), 4),
    }


def _normalize_full_pattern(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Return raw pattern with ISO timestamps and basic keys."""
    base = raw.copy()
    for key, val in list(base.items()):
        if "time" in key or "date" in key:
            try:
                base[key] = pd.to_datetime(val, unit="s", utc=True).strftime("%Y-%m-%dT%H:%M:%SZ")
            except Exception:
                pass
    norm = _patterns_from_finnhub([raw])[0] if raw else {}
    base["patternname"] = norm.get("pattern")
    base["patterntype"] = norm.get("direction")
    base["start_date"] = norm.get("start_date")
    base["end_date"] = norm.get("end_date")
    return base


def generate_all_timeframe_forecasts(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return forecasts and features for all timeframes in ``payload``."""

    output: Dict[str, Any] = {}

    for tf_key, block in payload.items():
        if "candles" not in block or "patterns" not in block:
            raise ValueError(f"timeframe {tf_key} missing required keys")

        df = _df_from_finnhub_candles(block["candles"])
        df = _normalize_ohlc(df)
        raw_patterns = block.get("patterns", [])
        simple_patterns = _patterns_from_finnhub(raw_patterns)

        features = build_feature_dict(df)

        if "daily" in tf_key or block.get("resolution") in {"D", "1d", "D1"}:
            pred = probabilistic_day_forecast(df[["open", "high", "low", "close"]], simple_patterns)
            forecast = {
                "direction": pred["bias"],
                "confidence": pred["confidence"],
                "prob_up": probability_blender([pred["prob_up"]]),
                "O": round(float(pred["ohlc"]["o"]), 2),
                "H": round(float(pred["ohlc"]["h"]), 2),
                "L": round(float(pred["ohlc"]["l"]), 2),
                "C": round(float(pred["ohlc"]["c"]), 2),
            }
        else:
            base_res = {"patterns": simple_patterns}
            tmp = refine_next_predictions(base_res, df)["next_prediction"]
            forecast = tmp
            forecast["prob_up"] = probability_blender([forecast["prob_up"]])

        norm_patterns = [_normalize_full_pattern(p) for p in raw_patterns]

        output[tf_key] = {
            "timeframe": block.get("resolution", tf_key),
            "next_prediction": forecast,
            "patterns": norm_patterns,
            "feature_stack": features,
            "support_resistance": {"levels": block.get("support_resistance", {}).get("levels", [])},
            "aggregate_indicator": block.get("aggregate_indicator", {}),
            "technical_indicators": block.get("technical_indicators", {}),
        }

    return output


if __name__ == "__main__":
    from pprint import pprint

    SAMPLE_MULTI = {
        "fintech_daily": {
            "resolution": "D",
            "candles": {
                "o": [210.0, 211.0],
                "h": [212.0, 212.5],
                "l": [209.5, 210.0],
                "c": [211.0, 212.0],
                "v": [1000, 1200],
                "t": [1718323200, 1718409600],
            },
            "patterns": [
                {
                    "patternname": "Double Bottom",
                    "patterntype": "bullish",
                    "atime": 1718323200,
                    "dtime": 1718409600,
                    "entry": 211.0,
                    "stoploss": 209.0,
                    "profit1": 215.0,
                    "profit2": 0,
                    "status": "complete",
                    "terminal": 0,
                    "mature": 1,
                }
            ],
            "support_resistance": {"levels": [208.0, 214.0]},
            "aggregate_indicator": {},
            "technical_indicators": {},
        },
        "fintech_hourly": {
            "resolution": "60",
            "candles": {
                "o": [211.0, 211.5, 212.0],
                "h": [211.5, 212.0, 212.5],
                "l": [210.5, 211.0, 211.5],
                "c": [211.4, 211.8, 212.2],
                "v": [300, 250, 400],
                "t": [1718398800, 1718402400, 1718406000],
            },
            "patterns": [],
            "support_resistance": {"levels": [210.0, 213.0]},
            "aggregate_indicator": {},
            "technical_indicators": {},
        },
    }

    result = generate_all_timeframe_forecasts(SAMPLE_MULTI)
    pprint(result)


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
        patterns = _patterns_from_finnhub(data.get("patterns", []))   # ← NEW
        return cdl.get("h", []), cdl.get("l", []), cdl.get("c", []), patterns

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

    def _mini_forecast(high: List[float], low: List[float], close: List[float], patterns: List[Dict[str, Any]] | None = None) -> Dict[str, float]:
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
        "1min": _extract(payload.get("fintech_one_minute")),
        "15min": _extract(payload.get("fintech_fifteen_minutes")),
        "1h": _extract(payload.get("fintech_hourly")),
        "1d": _extract(payload.get("fintech_daily")),
        "1w": _extract(payload.get("fintech_weekly")),
    }

    forecasts: Dict[str, Dict[str, float]] = {}
    for tf, series in tf_data.items():
        if series is None:
            continue
        h, l, c, patts = series
        if len(c) < 2:
            continue
        forecasts[tf] = _mini_forecast(h, l, c, patts)

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



