import pandas as pd
import numpy as np
import math
import functools
import io
from typing import Dict, Any, List, Optional, Tuple

# ────────────────────────────────────────────────────────────────
# 1)  REAL-WORLD OHLC: Historical NVDA data up to June 25
#    – real data from Yahoo Finance for forecasting June 26
# ────────────────────────────────────────────────────────────────
csv = """\
Date,Open,High,Low,Close,Volume
2025-04-21,103.08,106.50,102.75,106.43,220815041
2025-04-22,104.12,107.45,103.85,106.79,231002134
2025-04-23,104.52,104.80,102.02,102.71,247525971
2025-04-24,105.30,106.54,103.11,106.43,220815041
2025-04-25,106.85,111.92,105.73,111.00,251064672
2025-04-28,109.69,110.37,106.02,108.73,207708479
2025-04-29,107.67,110.20,107.44,109.02,170444263
2025-04-30,104.47,108.92,104.08,108.92,235044611
2025-05-01,113.08,114.94,111.30,111.61,236121507
2025-05-02,114.18,115.40,113.37,114.50,190194778
2025-05-05,112.91,114.67,112.66,113.82,133163241
2025-05-06,111.48,114.74,110.82,113.54,158525621
2025-05-07,113.05,117.68,112.28,117.06,207827821
2025-05-08,118.25,118.68,115.85,117.36,198428122
2025-05-09,117.35,118.23,115.21,116.65,132972189
2025-05-12,121.97,123.00,120.28,123.00,225023345
2025-05-13,124.98,131.22,124.47,129.93,330430105
2025-05-14,133.20,135.44,131.68,135.34,281180830
2025-05-15,134.30,136.30,132.66,134.83,226632563
2025-05-16,136.22,136.35,133.46,135.40,226542451
2025-05-19,132.39,135.87,132.39,135.57,193154571
2025-05-20,134.29,134.58,132.62,134.38,161514247
2025-05-21,133.06,137.40,130.59,131.80,270608700
2025-05-22,132.22,134.24,131.55,132.82,187344000
2025-05-23,129.99,132.67,129.15,131.28,198821300
2025-05-27,134.14,135.65,133.30,135.49,192953600
2025-05-28,136.02,137.24,134.78,134.80,304021100
2025-05-29,142.24,143.48,137.90,139.18,369241900
2025-05-30,138.71,139.61,132.91,135.12,333170900
2025-06-02,135.48,138.11,135.39,137.37,197663100
2025-06-03,138.77,141.99,137.94,141.21,225578800
2025-06-04,142.18,142.38,139.53,141.92,167120800
2025-06-05,142.16,143.99,138.82,139.98,231397900
2025-06-06,142.50,143.26,141.50,141.71,153986200
2025-06-09,143.18,144.98,141.93,142.62,185114500
2025-06-10,142.68,144.27,141.52,143.96,155881900
2025-06-11,144.61,144.99,141.87,142.83,167694000
2025-06-12,141.97,145.00,141.85,145.00,162365000
2025-06-13,142.48,143.58,140.85,141.97,180820600
2025-06-16,143.35,146.18,143.20,144.69,183133700
2025-06-17,144.49,145.22,143.78,144.12,139108000
2025-06-18,144.01,145.65,143.12,145.48,161494100
2025-06-20,145.45,146.20,142.65,143.85,242956200
2025-06-23,142.50,144.78,142.03,144.17,154308900
2025-06-24,145.56,147.96,145.50,147.90,187566100
2025-06-25,149.27,154.45,149.26,154.31,269146500
"""
#
# Open: 155.98
# •	High: 156.72
# •	Low: 154.00
# •	Close: 155.02
#
ohlc_df = pd.read_csv(io.StringIO(csv), parse_dates=["Date"])

# ────────────────────────────────────────────────────────────────
# 2)  ACTIVE PATTERNS LIST  (normally produced by earlier steps)
# ────────────────────────────────────────────────────────────────
# These patterns were identified in NVDA's price action leading up to June 25
active_patterns = [
  {"name": "Double Bottom",  "direction": "bullish"},
  {"name": "Engulfing",      "direction": "bullish"},
  {"name": "Doji",           "direction": "bullish"},
  {"name": "Support Level",  "direction": "bullish"},
]

# ────────────────────────────────────────────────────────────────
# 3) PATTERN RELIABILITY DATA (simplified for this test)
# ────────────────────────────────────────────────────────────────
def get_pattern_reliability():
    """Return a dictionary of pattern reliability scores."""
    return {
        "Double Bottom": 0.65,
        "Engulfing": 0.58,
        "Doji": 0.52,
        "Support Level": 0.60,
        "Head and Shoulders": 0.45,
    }

# ────────────────────────────────────────────────────────────────
# 4) FORECASTING FUNCTIONS (simplified from forecasting.py)
# ────────────────────────────────────────────────────────────────
def _normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lowercase."""
    df = df.rename(columns={c: c.lower() for c in df.columns})
    if "close" not in df.columns:
        for alt in ("adj close", "adjclose", "adjusted_close"):
            if alt in df.columns:
                df["close"] = df[alt]
                break
        else:
            raise KeyError("'close' column not found in OHLC data")
    return df

def _normalize_pattern_df(df: Any) -> pd.DataFrame:
    """Normalize pattern dataframe."""
    if df is None:
        return pd.DataFrame(columns=["name", "direction"])

    if isinstance(df, pd.DataFrame):
        work = df.copy()
    elif isinstance(df, (list, tuple, set)):
        if not df:
            return pd.DataFrame(columns=["name", "direction"])
        work = pd.DataFrame(list(df))
    elif isinstance(df, dict):
        work = pd.DataFrame(df)
    else:
        raise TypeError(f"Unsupported patterns container type: {type(df).__name__}")

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

    if "direction" not in work.columns:
        work["direction"] = "unknown"

    if "name" not in work.columns:
        raise KeyError("Pattern DataFrame lacks required column 'name'")

    work["direction"] = work["direction"].astype(str).str.lower().str.strip()

    ordered_cols = ["name", "direction"] + [c for c in work.columns if c not in ("name", "direction")]
    return work[ordered_cols]

@functools.lru_cache(maxsize=1)
def _load_pattern_stats() -> pd.DataFrame:
    """Load pattern statistics."""
    reliab = get_pattern_reliability()
    records = []
    for pat, r in reliab.items():
        r = max(0.01, min(0.99, r))  # clamp
        records.append((pat, "bullish", r))
        records.append((pat, "bearish", 1.0 - r))
    df = pd.DataFrame(records, columns=["pattern", "direction", "p"])
    df.set_index(["pattern", "direction"], inplace=True)
    return df

def probabilistic_day_forecast(ohlc_df: pd.DataFrame,
                              active_patterns: List[Dict[str, Any]],
                              num_mc_paths: int = 1000,
                              atr_period: int = 14,
                              beta_k: float = 1.0) -> Dict[str, Any]:
    """Generate a probabilistic forecast for the next trading day."""
    # Normalize data
    ohlc_df = _normalize_ohlc(ohlc_df)
    active_patterns = _normalize_pattern_df(active_patterns)

    stats = _load_pattern_stats()
    last_close = ohlc_df["close"].iloc[-1]

    # Combine pattern odds
    if active_patterns.empty:
        prob_up = 0.5
    else:
        log_odds_sum = 0.0
        dampening_factor = 0.7
        pattern_count = len(active_patterns)

        for _, p in active_patterns.iterrows():
            key = (p["name"], p["direction"])
            p_prob = stats["p"].get(key, 0.55)  # default mild edge
            p_prob = max(0.01, min(0.99, p_prob))
            sign = +1 if p["direction"] == "bullish" else -1
            log_odds_sum += math.log(p_prob / (1 - p_prob)) * sign * dampening_factor / max(1, math.sqrt(pattern_count))
        prob_up = 1 / (1 + math.exp(-log_odds_sum))

    # Cap confidence
    confidence = min(0.9, abs(prob_up - 0.5) * 2.0)
    bias = "bullish" if prob_up > 0.55 else "bearish" if prob_up < 0.45 else "neutral"

    # Calculate ATR and drift
    tr = np.maximum(ohlc_df["high"] - ohlc_df["low"],
                   np.maximum((ohlc_df["high"] - ohlc_df["close"].shift()).abs(),
                             (ohlc_df["low"] - ohlc_df["close"].shift()).abs()))
    atr = tr.rolling(atr_period, min_periods=1).mean().iloc[-1]
    atr_pct = atr / last_close if last_close > 0 else 0.0

    mu = (prob_up - 0.5) * 2.0 * beta_k * atr_pct

    # Monte-Carlo simulation
    returns = np.log(ohlc_df["close"]).diff().dropna()
    if returns.empty or returns.std(ddof=0) == 0.0:
        returns = pd.Series(np.random.normal(0, 1e-4, size=50))

    sampled_cc = np.random.choice(returns, size=num_mc_paths, replace=True)
    sampled_cc = np.exp(sampled_cc + mu) - 1.0

    close_samples = last_close * (1.0 + sampled_cc)
    high_samples = np.maximum(last_close, close_samples) + np.random.uniform(0.1, 0.5, num_mc_paths) * atr
    low_samples = np.minimum(last_close, close_samples) - np.random.uniform(0.1, 0.5, num_mc_paths) * atr

    # Point estimates & interval
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
        "patterns": active_patterns["name"].tolist() if not active_patterns.empty else [],
    }

# ────────────────────────────────────────────────────────────────
# 5)  CALL THE FORECASTER TO PREDICT JUNE 26 BASED ON DATA UP TO JUNE 25
# ────────────────────────────────────────────────────────────────
mc_paths = 2_000  # same as your run_analysis default
day_fcast = probabilistic_day_forecast(
    ohlc_df=ohlc_df,
    active_patterns=active_patterns,
    num_mc_paths=mc_paths,
    beta_k=1.0
)

ohlc = day_fcast["ohlc"]
print(f"\n🔮 Probabilistic forecast for NVDA on June 26, 2023 → bias: {day_fcast['bias']}, "
      f"P(up)={day_fcast['prob_up']:.2f}, conf={day_fcast['confidence']:.0%}\n"
      f"    O={ohlc['o']:.2f}  H={ohlc['h']:.2f}  "
      f"L={ohlc['l']:.2f}  C={ohlc['c']:.2f}"
      f"  (80 % interval: {day_fcast['interval_80']})")
