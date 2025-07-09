import functools
import math
from typing import Dict, Any, List, Optional
from typing import Tuple

import numpy as np
import pandas as pd

from .utils import get_pattern_reliability


# forecasting.py
# -------------
# Functions for forecasting based on pattern analysis


def _breakout_confirmed(df: pd.DataFrame, pattern: Dict) -> bool:
  """
  Check if a pattern breakout is confirmed by subsequent price action.

  Args:
      df: DataFrame with OHLC data
      pattern: Pattern dictionary

  Returns:
      Boolean indicating if breakout is confirmed
  """
  # Find the end date of the pattern in the dataframe
  try:
    end_idx = df[df['Date'] == pattern['end_date']].index[0]
  except (IndexError, KeyError):
    return False

  # Skip if we're at the end of the dataframe
  if end_idx >= len(df) - 1:
    return False

  # Get the next few bars after pattern completion
  next_bars = df.iloc[end_idx + 1:min(end_idx + 6, len(df))]

  if next_bars.empty:
    return False

  # Check for breakout confirmation based on pattern direction
  if pattern['direction'] == 'bullish':
    # For bullish patterns, check if price moves above the pattern high
    pattern_high = df.loc[df['Date'] == pattern['end_date'], 'High'].iloc[0]
    return any(next_bars['High'] > pattern_high * 1.01)  # 1% above pattern high

  elif pattern['direction'] == 'bearish':
    # For bearish patterns, check if price moves below the pattern low
    pattern_low = df.loc[df['Date'] == pattern['end_date'], 'Low'].iloc[0]
    return any(next_bars['Low'] < pattern_low * 0.99)  # 1% below pattern low

  return False


def _label_patterns(df: pd.DataFrame, patterns: List[Dict]) -> pd.DataFrame:
  """
  Add pattern labels to the dataframe.

  Args:
      df: DataFrame with OHLC data
      patterns: List of pattern dictionaries

  Returns:
      DataFrame with pattern labels
  """
  # Create a copy of the dataframe
  labeled_df = df.copy()

  # Add pattern columns
  labeled_df['Pattern'] = ''
  labeled_df['Pattern_Direction'] = ''
  labeled_df['Pattern_Score'] = 0.0

  # Label each pattern
  for pattern in patterns:
    try:
      start_idx = labeled_df[labeled_df['Date'] == pattern['start_date']].index[
        0]
      end_idx = labeled_df[labeled_df['Date'] == pattern['end_date']].index[0]

      # Label all rows in the pattern range
      for idx in range(start_idx, end_idx + 1):
        if labeled_df.loc[idx, 'Pattern'] == '':
          labeled_df.loc[idx, 'Pattern'] = pattern['pattern']
          labeled_df.loc[idx, 'Pattern_Direction'] = pattern['direction']
          labeled_df.loc[idx, 'Pattern_Score'] = pattern.get('value', 0)
        else:
          # If there's already a pattern, append the new one
          labeled_df.loc[idx, 'Pattern'] += f", {pattern['pattern']}"
          labeled_df.loc[
            idx, 'Pattern_Direction'] += f", {pattern['direction']}"
          labeled_df.loc[idx, 'Pattern_Score'] = max(
              labeled_df.loc[idx, 'Pattern_Score'], pattern.get('value', 0))
    except (IndexError, KeyError):
      continue

  return labeled_df


def _decay(t_delta_h: float, half_life_h: float) -> float:
  """Exponential decay factor."""
  return math.exp(-t_delta_h / half_life_h)


# ---------------------------------------------------------------------------
# Refined next-bar forecaster with pattern-target & IV-aware High
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
    iv_move: float | None = None,          # $-move of front-day straddle
    premarket_high: float | None = None,   # optional pre-market print
) -> Dict[str, Any]:
  """
  One-bar OHLC forecast.

  • High now blends three components
      1.  Historical up-excursion quantile (as before)
      2.  Pattern breakout height (0.8 × max bullish height)
      3.  IV-scaled volatility multiplier

  Everything else (Open, Low, Close) stays as in the previous version.
  """
  if abs(weight_pattern + weight_volatility - 1.0) > 1e-6:
    raise ValueError("weights must sum to 1")

  out = results.copy()
  if not results.get("patterns") or len(df) < atr_window + 5:
    return out

  latest = df.iloc[-1]
  close  = latest["Close"]
  now_ts = pd.to_datetime(latest["Date"])

  # ── 1.  EW-ATR & excursion series ──────────────────────────────────
  tr = np.maximum(
      df["High"] - df["Low"],
      np.maximum((df["High"] - df["Close"].shift()).abs(),
                 (df["Low"]  - df["Close"].shift()).abs()),
      )
  atr_series = tr.ewm(alpha=1/atr_window, adjust=False).mean()
  atr = atr_series.iloc[-1]

  up_ex = (df["High"] - df[["Open", "Close"]].max(axis=1)) / atr_series
  dn_ex = (df[["Open", "Close"]].min(axis=1) - df["Low"])  / atr_series
  up_move = (df["Close"] - df["Open"]) / atr_series
  dn_move = (df["Open"] - df["Close"]) / atr_series
  pos_mask = up_move >= 0
  neg_mask = dn_move >= 0

  # ── 2.  Pattern-derived log-odds (unchanged) ───────────────────────
  reliab = get_pattern_reliability()
  log_odds = 0.0
  for p in results["patterns"]:
    decay = math.exp(-(now_ts - pd.to_datetime(p["end_date"])).total_seconds()/3600 / half_life_h)
    base  = max(0.01, min(0.99, reliab.get(p["pattern"], 0.55)))
    strength = (p.get("value", 50)/100)
    sign = +1 if p["direction"] == "bullish" else -1
    log_odds += sign * math.log(base/(1-base)) * strength * decay

  prob_up  = 1/(1+math.exp(-log_odds))
  direction = "bullish" if prob_up>0.55 else "bearish" if prob_up<0.45 else "neutral"
  conf = abs(prob_up-0.5)*2

  def _q(base: float, span: float = .3):   # confidence-dependent quantile
    return np.clip(base + span*conf, .05, .95)

  # ── 3.  Vol-multiplier from implied move ---------------------------
  vol_scale = 1.0
  if iv_move and iv_move>0:
    iv_pct = iv_move / close
    atr_pct = atr / close
    vol_scale = np.clip(0.6 + 0.4 * (iv_pct / max(1e-6, atr_pct)), 0.6, 1.5)

  # ── 4.  Pattern breakout target (bullish) --------------------------
  max_bull_height = max((p.get("height",0) for p in results["patterns"]
                         if p["direction"]=="bullish"), default=0.0)

  # ── 5.  Price targets ---------------------------------------------
  if direction == "bullish":
    hi_exc   = np.nanquantile(up_ex[pos_mask], _q(.55)) if pos_mask.any() else hi_lo_q_hi
    cl_mv    = np.nanquantile(up_move[pos_mask], _q(.50,.25)) if pos_mask.any() else .5
    base_high = close + hi_exc * atr * weight_volatility * vol_scale
    pattern_high = close + 0.8 * max_bull_height
    high = max(base_high, pattern_high)
    low  = close - np.nanquantile(dn_ex, hi_lo_q_lo) * atr * .5
    close_next = close + cl_mv * atr * weight_pattern

  elif direction == "bearish":
    hi_exc = np.nanquantile(up_ex[neg_mask], 1-_q(.55)) if neg_mask.any() else hi_lo_q_lo
    base_high = close + hi_exc * atr * 0.4 * vol_scale
    high = base_high
    low  = close - np.nanquantile(dn_ex[neg_mask], _q(.55)) * atr * weight_volatility
    cl_mv = np.nanquantile(dn_move[neg_mask], _q(.50,.25)) if neg_mask.any() else .5
    close_next = close - cl_mv * atr * weight_pattern
  else:
    hi_exc = np.nanquantile(up_ex, .5)
    dn_exc = np.nanquantile(dn_ex, .5)
    high = close + hi_exc * atr * .5 * vol_scale
    low  = close - dn_exc * atr * .5
    close_next = close

  # pre-market high guard-rail
  if premarket_high:
    high = max(high, premarket_high)

  # ── 6.  Gap / open (unchanged) ------------------------------------
  gaps = ((df["Open"] / df["Close"].shift()) - 1).dropna()
  gap_pct = np.random.choice(gaps.values) if not gaps.empty else np.random.normal(0, 0.002)
  open_next = close * (1 + gap_pct)

  high = max(high, open_next, close_next)
  low  = min(low,  open_next, close_next)

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
        {'open': opens, 'high': highs, 'low': lows, 'close': closes}).corr()

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
      hl_range = np.random.normal(self.mean_hl_range,
                                  self.high_low_ranges.std())
      hl_range = max(0.005, hl_range)  # Ensure positive range

      # Predict open-close range
      oc_range = np.random.normal(self.mean_oc_range,
                                  self.open_close_ranges.std())
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
          {'O': pred_open, 'H': pred_high, 'L': pred_low, 'C': pred_close})

      # Update last close for next iteration
      last_close = pred_close

    return predictions[0] if days == 1 else predictions


def build_feature_stack(df_hist: pd.DataFrame,
    df_today_min: Optional[pd.DataFrame], daily_patterns: List[Dict],
    intraday_patterns: List[Dict], vwap_trend: str,
    morning_cutoff: str = "10:30") -> Dict[str, Any]:
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

    bullish = sum(1 for p in patts if p['direction'] == 'bullish')
    bearish = sum(1 for p in patts if p['direction'] == 'bearish')

    if bullish == bearish:
      return 0.0

    # Calculate normalized score between -1 and 1
    total = bullish + bearish
    return (bullish - bearish) / total

  # Historical features
  if not df_hist.empty:
    # Price momentum
    features['price_momentum'] = df_hist['Close'].pct_change(5).iloc[-1]

    # Volatility
    features['volatility'] = \
      df_hist['High'].sub(df_hist['Low']).div(df_hist['Close']).rolling(
          10).mean().iloc[-1]

    # Daily pattern bias
    features['daily_pattern_bias'] = _bias_score(daily_patterns)

    # Recent performance
    features['week_return'] = df_hist['Close'].pct_change(5).iloc[-1]
    features['month_return'] = df_hist['Close'].pct_change(20).iloc[-1]

  # Intraday features
  if df_today_min is not None and not df_today_min.empty:
    # Morning vs. full day performance
    morning_data = df_today_min[
      df_today_min['Date'].str.contains(morning_cutoff, regex=False)]

    if not morning_data.empty:
      morning_open = morning_data.iloc[0]['Open']
      morning_close = morning_data.iloc[-1]['Close']
      features['morning_return'] = (morning_close / morning_open) - 1

    # Intraday pattern bias
    features['intraday_pattern_bias'] = _bias_score(intraday_patterns)

    # Intraday volatility
    features['intraday_volatility'] = df_today_min['High'].max() / df_today_min[
      'Low'].min() - 1

    # VWAP trend
    features['vwap_trend'] = 1 if vwap_trend == "UP" else -1

  # Combined features
  features['combined_bias'] = (
      features.get('daily_pattern_bias', 0) * 0.6 + features.get(
      'intraday_pattern_bias', 0) * 0.4)

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


# ---------------------------------------------------------------------------
# Better Monte-Carlo forecaster – state-scaled bootstrap + pattern targets
# ---------------------------------------------------------------------------
def probabilistic_day_forecast(
    ohlc_df: pd.DataFrame,
    active_patterns: List[Dict[str, Any]],
    *,
    num_mc_paths: int = 1000,
    atr_period: int = 14,
    beta_k: float = 1.0,
    bootstrap_block: int = 5,        # consecutive-day block length
) -> Dict[str, Any]:
  """
  Probabilistic OHLC forecast for the *next* regular-hours daily bar.

  Improvements vs. v1
  -------------------
  • **Block bootstrap** keeps short-term autocorr / volatility clustering.
  • **Vol-regime scaling** rescales historical returns to the current 20-day σ.
  • **Pattern price targets** blended in (height × 0.8) ⇒ fatter bullish/bear tails.
  • **Skew-aware point close** – mean if drift |μ| > 0, median otherwise.
  """
  # 0. Normalise inputs -------------------------------------------------
  ohlc_df       = _normalize_ohlc(ohlc_df)
  active_pats   = _normalize_pattern_df(active_patterns)
  stats         = _load_pattern_stats()
  last_close    = ohlc_df["close"].iloc[-1]

  # 1. Pattern-derived probability -------------------------------------
  if active_pats.empty:
    prob_up = 0.5
  else:
    damp    = 0.7
    log_odds = 0.0
    for _, p in active_pats.iterrows():
      p_prob = stats["p"].get((p["name"], p["direction"]), 0.55)
      p_prob = np.clip(p_prob, 0.01, 0.99)
      sign   = +1 if p["direction"] == "bullish" else -1
      log_odds += sign * np.log(p_prob / (1 - p_prob)) * damp
    prob_up = 1 / (1 + np.exp(-log_odds))

  confidence = min(0.9, abs(prob_up - 0.5) * 2)
  bias = "bullish" if prob_up > 0.55 else "bearish" if prob_up < 0.45 else "neutral"

  # 2. Volatility & drift ----------------------------------------------
  tr   = np.maximum(
      ohlc_df["high"] - ohlc_df["low"],
      np.maximum((ohlc_df["high"] - ohlc_df["close"].shift()).abs(),
                 (ohlc_df["low"]  - ohlc_df["close"].shift()).abs()))
  atr  = tr.rolling(atr_period, min_periods=1).mean().iloc[-1]
  atr_pct = atr / last_close
  mu   = (prob_up - 0.5) * 2 * beta_k * atr_pct

  # 3. Block bootstrap with regime scaling -----------------------------
  log_ret = np.log(ohlc_df["close"]).diff().dropna()
  if log_ret.empty:
    log_ret = pd.Series(np.random.normal(0, 1e-4, size=50))

  # scale returns to current vol regime
  sigma_hist = log_ret.std(ddof=0)
  sigma_curr = log_ret.tail(20).std(ddof=0)
  scaler     = sigma_curr / sigma_hist if sigma_hist > 0 else 1.0

  # build block-bootstrapped sample
  pool = []
  while len(pool) < num_mc_paths:
    i = np.random.randint(0, len(log_ret) - bootstrap_block)
    pool.extend(log_ret.iloc[i:i + bootstrap_block].values)
  sampled = np.array(pool[:num_mc_paths]) * scaler
  sampled = np.exp(sampled + mu) - 1
  close_samples = last_close * (1 + sampled)

  # 4. Blend in pattern price targets ----------------------------------
  tgt_prices = []
  for _, p in active_pats.iterrows():
    h = p.get("height", 0)
    if h:
      sign = +1 if p["direction"] == "bullish" else -1
      tgt_prices.append(last_close + sign * 0.8 * h)
  if tgt_prices:
    close_samples = np.concatenate([close_samples, tgt_prices])

  # 5. Build high/low samples (directional) ----------------------------
  high_samples = np.maximum(last_close, close_samples) + np.random.uniform(0.1, 0.5, len(close_samples)) * atr
  low_samples  = np.minimum(last_close, close_samples) - np.random.uniform(0.1, 0.5, len(close_samples)) * atr

  # 6. Point estimates & interval --------------------------------------
  open_ = last_close
  close_ = float(np.mean(close_samples) if abs(mu) > 0 else np.median(close_samples))
  high_  = float(np.quantile(high_samples, 0.75))
  low_   = float(np.quantile(low_samples, 0.25))
  p10, p90 = np.quantile(close_samples, [0.10, 0.90])

  return {
    "bias": bias,
    "prob_up": round(float(prob_up), 4),
    "confidence": round(float(confidence), 4),
    "expected_return": round(float(mu), 4),
    "ohlc": {"o": open_, "h": high_, "l": low_, "c": close_},
    "interval_80": (round(float(p10), 4), round(float(p90), 4)),
    "patterns": active_pats["name"].tolist() if not active_pats.empty else [],
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
          "'close' column not found in OHLC data (even after normalising)")

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
  import \
    pandas as pd  # local import keeps the signature usable in typing-only contexts

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
        "Pattern DataFrame lacks required column 'name' after normalisation")

  # standardise textual content
  work["direction"] = (
    work["direction"].astype(str, copy=False).str.lower().str.strip())

  # keep original extra columns (if any) after the canonical two
  ordered_cols = ["name", "direction"] + [c for c in work.columns if
    c not in ("name", "direction")]
  return work[ordered_cols]
