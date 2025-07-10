"""
today_forecast.py
-----------------
Derive an OHLC forecast for the *current* U.S. trading day
given:

    • daily history               (stock_data_daily)
    • partial-day hourly bars     (stock_data_hourly)   up to “now”
    • detected patterns           (patterns)

Call `make_today_forecast(json_blob)` at 15:00 EET every day.
"""

from __future__ import annotations
import json, math, random, datetime as dt
from typing import Any, Dict, List

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# 1.  Helpers
# ----------------------------------------------------------------------
def _to_df(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Fast dict-list → DataFrame → enforce datetime (mixed formats)."""
    df = pd.DataFrame(records)
    # Robustly parse both pure dates and date+time strings:
    df["Date"] = pd.to_datetime(
        df["Date"],
        format="mixed",  # supports 'YYYY-MM-DD' and 'YYYY-MM-DD HH:MM'
        infer_datetime_format=True,  # speed up parsing by inferring
    )
    return df.sort_values("Date", ignore_index=True)


def _ew_atr(df: pd.DataFrame, span: int = 14) -> float:
    tr = np.maximum(
        df["High"] - df["Low"],
        np.maximum(
            abs(df["High"] - df["Close"].shift()), abs(df["Low"] - df["Close"].shift())
        ),
    )
    return tr.ewm(span=span, adjust=False).mean().iloc[-1]


def _pattern_log_odds(patterns: pd.DataFrame) -> float:
    """Reliability-weighted, time-decayed log-odds."""
    # simple static reliability map
    rel = {
        "Double Bottom": 0.70,
        "Double Top": 0.30,
        "Bullish Harami": 0.58,
        "Bearish Harami": 0.42,
        "Inverted Hammer": 0.55,
        "Evening Star": 0.35,
    }
    now = pd.Timestamp.now(tz="UTC")
    half_life_h = 8.0
    lodds = 0.0
    for _, p in patterns.iterrows():
        base = rel.get(p["pattern"], 0.52)
        end_ts = pd.to_datetime(p["end_date"], utc=True)
        decay = math.exp(-((now - end_ts).total_seconds() / 3600) / half_life_h)
        strength = p.get("value", 0.5)
        part = math.log(base / (1 - base)) * strength * decay
        lodds += part if p["direction"] == "bullish" else -part
    return lodds


# ----------------------------------------------------------------------
# 2.  Main public API
# ----------------------------------------------------------------------
def make_today_forecast(input_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parameters
    ----------
    input_json  – the large dict you pasted (daily bars, hourly bars, patterns)

    Returns
    -------
    dict with
        direction, prob_up, confidence, O/H/L/C (for *today’s* close),
        interval_80  (10–90 % band for the close)
    """

    df_daily = _to_df(input_json["stock_data_daily"])
    df_hourly = _to_df(input_json["stock_data_hourly"])
    patterns = pd.DataFrame(input_json["patterns"])

    # ------------------------------------------------------------------
    # a) intraday state
    # ------------------------------------------------------------------
    last_hour = df_hourly.iloc[-1]  # 14:30 ET bar
    last_close = last_hour["Close"]
    today_open = (
        df_hourly[df_hourly["Date"].dt.hour == 15].iloc[0]["Open"]
        if not df_hourly.empty
        else df_daily.iloc[-1]["Open"]
    )

    # realised intraday σ so far (std of 1-h returns)
    intraday_ret = df_hourly["Close"].pct_change().dropna()
    rv = intraday_ret.std(ddof=0) if not intraday_ret.empty else 0.002

    # remaining trading hours (until 16:00 ET)
    hours_left = max(0, 16 - last_hour["Date"].hour - 1)

    # ------------------------------------------------------------------
    # b) pattern bias
    # ------------------------------------------------------------------
    lodds = _pattern_log_odds(patterns)
    prob_up = 1 / (1 + math.exp(-lodds))
    dir_tag = (
        "bullish" if prob_up > 0.55 else "bearish" if prob_up < 0.45 else "neutral"
    )
    conf = round(abs(prob_up - 0.5) * 2, 2)

    # ------------------------------------------------------------------
    # c) volatility yard-stick for the *remaining* day
    # ------------------------------------------------------------------
    daily_atr = _ew_atr(df_daily)
    rem_sig = math.sqrt(hours_left / 6.5) * rv * last_close + 0.3 * daily_atr
    rem_sig_pct = rem_sig / last_close

    drift = (prob_up - 0.5) * 0.5 * rem_sig_pct  # µ bounded to ±½ σ

    # Monte-Carlo 1000 paths of remaining-day return
    paths = np.random.normal(drift, rem_sig_pct, size=1000)
    close_samples = last_close * (1 + paths)
    p10, p90 = np.quantile(close_samples, [0.10, 0.90])

    close_pt = float(np.mean(close_samples) if drift else np.median(close_samples))
    high_pt = float(max(close_pt, last_close) + 0.25 * rem_sig)
    low_pt = float(min(close_pt, last_close) - 0.25 * rem_sig)

    return {
        "direction": dir_tag,
        "prob_up": round(prob_up, 3),
        "confidence": conf,
        "O": round(float(today_open), 2),
        "H": round(high_pt, 2),
        "L": round(low_pt, 2),
        "C": round(close_pt, 2),
        "interval_80": [round(float(p10), 2), round(float(p90), 2)],
    }


# ----------------------------------------------------------------------
# 3.  CLI quick test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import sys, pathlib, json

    blob = json.loads(pathlib.Path(sys.argv[1]).read_text())
    print(json.dumps(make_today_forecast(blob), indent=2))
