from __future__ import annotations

"""Utilities for gathering intraday forecasting factors."""

from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict

import json
import os
import requests

import pandas as pd
import numpy as np
import yfinance as yf
from textblob import TextBlob

# Optional imports that may not be installed by default
try:
    from newsapi import NewsApiClient
except Exception:  # pragma: no cover - optional dependency
    NewsApiClient = None  # type: ignore


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_implied_move(symbol: str) -> dict:
    """Return the implied absolute and percent move from the ATM straddle."""
    ticker = yf.Ticker(symbol)
    try:
        expiries = ticker.options
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch option chain: {exc}")
    if not expiries:
        raise RuntimeError("No option expiries returned")
    expiry = expiries[0]
    try:
        chain = ticker.option_chain(expiry)
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch option chain for {expiry}: {exc}")
    last_close = ticker.history(period="1d")["Close"].iloc[-1]
    calls = chain.calls
    puts = chain.puts
    if calls.empty or puts.empty:
        raise RuntimeError("Option chain missing call/put data")
    calls_idx = (calls["strike"] - last_close).abs().argsort().iloc[0]
    puts_idx = (puts["strike"] - last_close).abs().argsort().iloc[0]
    price = float(calls.loc[calls_idx, "lastPrice"] + puts.loc[puts_idx, "lastPrice"])
    return {"absolute": round(price, 2), "percent": round(price / last_close * 100, 2)}


def get_opening_microstructure(symbol: str, date: datetime) -> dict:
    """Return auction imbalance and VWAP trend for the first hour of trading."""
    # --- Opening auction imbalance (requires Polygon) ---------------------
    key = os.getenv("POLYGON_KEY")
    if not key:
        raise RuntimeError("POLYGON_KEY not set for opening auction data")
    url = (
        f"https://api.polygon.io/v1/open-close/{symbol}/{date.strftime('%Y-%m-%d')}?apiKey={key}&unadjusted=true"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch opening data: {exc}")
    imbalance_shares = data.get("imbalance", 0)
    imbalance_dollars = data.get("imbalance", 0) * data.get("open", 0)

    # --- First hour VWAP trend -------------------------------------------
    start = date.strftime("%Y-%m-%d")
    end = (date + timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        df = yf.download(symbol, start=start, end=end, interval="1m", progress=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to download intraday data: {exc}")
    if df.empty:
        raise RuntimeError("No intraday data returned")
    df = df.between_time("09:30", "10:30")
    if df.empty:
        raise RuntimeError("No data for first hour")
    vwap = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()
    slope = np.polyfit(range(len(vwap)), vwap, 1)[0]
    trend = "up" if slope > 0 else "down" if slope < 0 else "flat"
    return {
        "auction_imbalance_shares": int(imbalance_shares),
        "auction_imbalance_dollars": float(imbalance_dollars),
        "first_hour_vwap_trend": trend,
    }


def get_active_catalysts(symbol: str, date: datetime) -> List[dict]:
    """Return scheduled catalysts like earnings or macro prints for the date."""
    ticker = yf.Ticker(symbol)
    catalysts: List[dict] = []
    try:
        cal = ticker.calendar.T
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch calendar data: {exc}")
    if "Earnings Date" in cal.index:
        earnings_ts = cal.loc["Earnings Date"].iloc[0]
        if pd.isna(earnings_ts):
            pass
        else:
            when = pd.to_datetime(earnings_ts).tz_localize(None).isoformat() + "Z"
            if pd.to_datetime(when).date() == date.date():
                catalysts.append({"type": "earnings", "scheduled_time_utc": when, "expected_vs_prior": None})
    # Macro events placeholder
    return catalysts


def get_headline_sentiment(symbol: str) -> dict:
    """Return polarity score for recent major headlines."""
    key = os.getenv("NEWSAPI_KEY")
    if not key:
        raise RuntimeError("NEWSAPI_KEY not set for headline sentiment")
    if NewsApiClient is None:
        raise RuntimeError("newsapi-python not installed")
    client = NewsApiClient(api_key=key)
    now = datetime.utcnow()
    since = now - timedelta(hours=12)
    query = f"{symbol}"
    res = client.get_everything(q=query, from_param=since.isoformat(), to=now.isoformat(), language="en", sort_by="publishedAt", page_size=100)
    articles = res.get("articles", [])
    if not articles:
        raise RuntimeError("No headlines returned")
    scores = []
    for a in articles:
        text = f"{a.get('title','')} {a.get('description','')}"
        scores.append(TextBlob(text).sentiment.polarity)
    if not scores:
        raise RuntimeError("No headlines scored")
    avg = float(np.mean(scores))
    dominant = "bullish" if avg > 0.05 else "bearish" if avg < -0.05 else "neutral"
    return {"dominant": dominant, "score": round(avg, 2), "sample_size": len(scores)}


def get_confirmed_patterns(symbol: str) -> List[dict]:
    """Return recent confirmed candlestick or chart patterns."""
    try:
        df = yf.download(symbol, period="35d", interval="1d", progress=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to download history: {exc}")
    if df.empty:
        raise RuntimeError("No historical data")
    df.reset_index(inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    last30 = df.tail(30)
    from research_stocks.tools.pattern_analysis.pattern_analyzer import analyze_patterns  # type: ignore
    results = analyze_patterns(symbol, last30, last30, window=5)
    patterns = []
    cutoff = last30["Date"].iloc[-3]
    for p in results.get("patterns", []):
        end = pd.to_datetime(p.get("end_date"))
        if end >= cutoff:
            patterns.append({
                "name": p.get("pattern"),
                "direction": p.get("direction"),
                "strength_score": round(float(p.get("value", 0)), 2),
            })
    return patterns


def collect_intraday_factors(symbol: str, output_dir: Path) -> Path:
    """Collect all factors and export them as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.utcnow()
    data = {
        "symbol": symbol.upper(),
        "timestamp": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "implied_move": get_implied_move(symbol),
        "opening_microstructure": get_opening_microstructure(symbol, now),
        "active_catalysts": get_active_catalysts(symbol, now),
        "headline_sentiment": get_headline_sentiment(symbol),
        "confirmed_patterns": get_confirmed_patterns(symbol),
    }
    path = output_dir / f"intraday_forecast_inputs_{symbol.upper()}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect intraday forecast inputs")
    parser.add_argument("--symbols", required=True, help="Comma separated list of tickers")
    parser.add_argument("--output", required=True, help="Directory to save JSON files")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    out_dir = Path(args.output)
    for sym in symbols:
        try:
            path = collect_intraday_factors(sym, out_dir)
            print(f"Wrote {path}")
        except Exception as exc:
            print(f"Failed to collect factors for {sym}: {exc}")
