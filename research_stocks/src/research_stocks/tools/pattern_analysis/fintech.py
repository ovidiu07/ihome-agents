"""Finnhub pattern analysis wrapper."""

from __future__ import annotations

import json
import logging
import os
import requests
import time
import yfinance as yf
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from functools import lru_cache
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, RootModel
from typing import Any, Dict, Optional, List

logger = logging.getLogger(__name__)
# Reduce verbosity by defaulting to INFO level
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

BASE_URL = "https://finnhub.io/api/v1"
BACKOFF_FACTOR = 1.5
MAX_RETRIES = 3

# Load environment once on import
load_dotenv()

# Optional throttle delay between requests (seconds)
FINNHUB_THROTTLE = float(os.getenv("FINNHUB_THROTTLE", "0.4"))

# Track number of API calls for logging
_API_CALL_COUNT = 0

# Remember the timestamp of the last call for throttling
_LAST_CALL_TIME = 0.0

# Simple cache for technical indicators
_TECHNICAL_CACHE: dict[tuple, dict[str, Any]] = {}

# Default session used when none is provided
_DEFAULT_SESSION = requests.Session()


class QuoteResponse(BaseModel):
  """Real-time quote for a symbol."""
  c: float = Field(..., description="Current price")
  h: float = Field(..., description="High price of the day")
  l: float = Field(..., description="Low price of the day")
  o: float = Field(..., description="Open price of the day")
  pc: float = Field(..., description="Previous close price")
  t: str = Field(..., description="Timestamp of the quote, ISO8601 UTC")

  @field_validator("t", mode="before")
  def _to_iso(cls, v: int) -> str:
    # Convert UNIX timestamp (seconds) into ISO8601 UTC
    return datetime.utcfromtimestamp(v).isoformat() + "Z"


def get_quote(symbol: str,
    session: Optional[requests.Session] = None, ) -> QuoteResponse:
  """
  Return a real-time quote for `symbol`.
  https://finnhub.io/docs/api/quote
  """
  data = _call_finnhub("/quote", {"symbol": symbol.upper()}, session)
  return QuoteResponse.model_validate(data)


class CompanyNewsItem(BaseModel):
  """Single company news item from Finnhub API."""
  category: str  # Category, e.g., 'company'
  datetime: int | str  # UNIX timestamp or ISO8601 UTC string
  headline: str
  id: int
  image: Optional[str]
  related: str
  source: str
  summary: str
  url: str

  @field_validator("datetime", mode="before")
  def _normalize_datetime(cls, v: int | str) -> str:
    if isinstance(v, int):
      return datetime.utcfromtimestamp(v).isoformat() + "Z"
    return v


class CompanyNewsResponse(RootModel[List[CompanyNewsItem]]):
  """List of company news items."""


def get_company_news(symbol: str, start: datetime, end: datetime,
    session: Optional[requests.Session] = None, ) -> List[CompanyNewsItem]:
  """
  Return company news for `symbol` from `start` to `end`.
  https://finnhub.io/docs/api/company-news
  """
  params = {"symbol": symbol.upper(), "from": start.strftime("%Y-%m-%d"),
    "to": end.strftime("%Y-%m-%d"), }
  # data = _call_finnhub("/company-news", params, session)
  # resp = CompanyNewsResponse.model_validate(data)
  return []


def _get_token() -> str:
  """Return API token from environment or raise."""
  token = os.getenv("FINNHUB_TOKEN")
  if not token:
    raise ValueError("FINNHUB_TOKEN not set in environment")
  return token


def _call_finnhub(path: str, params: Dict[str, Any],
    session: Optional[requests.Session] = None, ) -> Any:
  """Perform a GET request with retries and backoff."""
  global _API_CALL_COUNT, _LAST_CALL_TIME

  token = _get_token()
  params = dict(params)
  params["token"] = token
  sess = session or _DEFAULT_SESSION

  delay = 1.0
  for attempt in range(1, MAX_RETRIES + 1):
    try:
      if FINNHUB_THROTTLE:
        elapsed = time.monotonic() - _LAST_CALL_TIME
        if elapsed < FINNHUB_THROTTLE:
          time.sleep(FINNHUB_THROTTLE - elapsed)
      url = f"{BASE_URL}{path}"
      # full_url = requests.Request('GET', url, params=params).prepare().url
      # logger.debug("Finnhub request: %s", full_url)
      resp = sess.get(url, params=params, timeout=20)
      # Gracefully handle “no data” (422) and server errors (500)
      if resp.status_code in (422, 500):
        return None

      if resp.status_code >= 400:
        raise requests.HTTPError(f"{resp.status_code} error: {resp.text}",
                                 response=resp)
      data = resp.json()
      if not data:
        raise ValueError("Empty response")
      _API_CALL_COUNT += 1
      _LAST_CALL_TIME = time.monotonic()
      return data
    except (requests.RequestException, ValueError) as exc:
      logger.warning("Request failed (%s/%s): %s", attempt, MAX_RETRIES, exc)
      if attempt == MAX_RETRIES:
        raise
      time.sleep(delay)
      delay *= BACKOFF_FACTOR

  raise RuntimeError("Unreachable retry loop")


class CandleResponse(BaseModel):
  """Candle data series."""
  o: list[float] = Field(..., description="Open prices")
  h: list[float] = Field(..., description="High prices")
  l: list[float] = Field(..., description="Low prices")
  c: list[float] = Field(..., description="Close prices")
  v: list[float] = Field(..., description="Volume values")
  t: list[str] = Field(..., description="ISO time stamps")
  s: str = Field(..., description="Response status")

  @field_validator("t", mode="before")
  def _to_iso(cls, v: list[int]) -> list[str]:
    return [datetime.utcfromtimestamp(ts).isoformat() + "Z" for ts in v]

  class Config:
    extra = "allow"


class PatternRecognitionResponse(BaseModel):
  """Detected chart patterns."""
  points: list[dict[str, Any]]

  class Config:
    extra = "allow"


class SupportResistanceResponse(BaseModel):
  """Support and resistance levels."""
  levels: list[float]

  class Config:
    extra = "allow"


class AggregateIndicatorResponse(BaseModel):
  """Aggregated indicator score."""
  trend: dict[str, Any]
  technicalAnalysis: dict[str, Any]

  class Config:
    extra = "allow"


class TechnicalIndicatorResponse(BaseModel):
  """Custom technical indicator data."""

  class Config:
    extra = "allow"


def get_candles(symbol: str, resolution: str, start: datetime, end: datetime,
    session: Optional[requests.Session] = None, ) -> CandleResponse:
  """Return OHLCV candles for ``symbol`` between ``start`` and ``end``."""
  params = {"symbol": symbol.upper(), "resolution": resolution,
    "from": int(start.timestamp()), "to": int(end.timestamp()), }
  logger.warning(
    f"No candle data returned for {symbol} between {start} and {end}, falling back to yfinance")
  # Map resolution to yfinance interval
  interval = f"{resolution}m" if resolution.isdigit() else (
    "1d" if resolution.upper() == "D" else "1wk")
  # Fetch from yfinance
  df = yf.Ticker(symbol).history(start=start, end=end, interval=interval,
                                 prepost=True)
  if df.empty:
    raise ValueError("No candle data available from yfinance fallback")
  df = df.tail(60)  # ✅ Keep only the most recent 60 candles
  # Build the same dict shape
  data = {"o": df["Open"].tolist(), "h": df["High"].tolist(),
    "l": df["Low"].tolist(), "c": df["Close"].tolist(),
    "v": df["Volume"].astype(float).tolist(),
    "t": [int(ts.timestamp()) for ts in df.index.to_pydatetime()], "s": "ok"}
  return CandleResponse.model_validate(data)


def get_pattern_recognition(symbol: str, resolution: str,
    session: Optional[requests.Session] = None, ) -> PatternRecognitionResponse:
  """Return detected classical patterns for ``symbol``."""
  data = _call_finnhub("/scan/pattern", {"symbol": symbol.upper(),
                                         "resolution": resolution.upper()},
                       session)
  return PatternRecognitionResponse.model_validate(data)


def get_support_resistance(symbol: str, resolution: str,
    session: Optional[requests.Session] = None, ) -> SupportResistanceResponse:
  """Return support and resistance levels for ``symbol``."""
  data = _call_finnhub("/scan/support-resistance", {"symbol": symbol.upper(),
                                                    "resolution": resolution.upper()},
                       session)
  return SupportResistanceResponse.model_validate(data)


def get_aggregate_indicator(symbol: str, resolution: str,
    session: Optional[requests.Session] = None, ) -> AggregateIndicatorResponse:
  """Return Finnhub's aggregate technical indicator for ``symbol``."""
  data = _call_finnhub("/scan/technical-indicator", {"symbol": symbol.upper(),
                                                     "resolution": resolution.upper()},
                       session)
  return AggregateIndicatorResponse.model_validate(data)


def get_technical_indicator(symbol: str, indicator: str, resolution: str = "D",
    start: datetime | None = None, end: datetime | None = None,
    session: Optional[requests.Session] = None, ) -> dict[str, Any] | None:
  """
  Return a single technical indicator series for ``symbol``.
  Only one API call is made per indicator name.
  Time-periods are automatically tuned for intraday vs daily.
  """
  if start is None:
    start = datetime.utcnow() - timedelta(days=365)
  if end is None:
    end = datetime.utcnow()

  cache_key = (symbol.upper(), indicator.lower(), resolution,
               int(start.timestamp()), int(end.timestamp()),)
  if cache_key in _TECHNICAL_CACHE:
    return _TECHNICAL_CACHE[cache_key]

  params = {"symbol": symbol.upper(), "indicator": indicator.lower(),
    "resolution": resolution, "from": int(start.timestamp()),
    "to": int(end.timestamp()), }

  # Determine if we're on an intraday chart
  intraday = resolution not in {"D", "W"}

  # Set ideal time-periods for each indicator
  ind = indicator.lower()
  # Aroon indicators default to a 3-period window
  if ind in {"aroon", "aroonosc"}:
    params["timeperiod"] = 3
  elif ind in {"sma", "wma", "dema", "tema", "trima", "kama", "t3"}:
    # Short MA for intraday, standard for daily
    params["timeperiod"] = 8 if intraday else 14

  elif ind == "ema":
    params["timeperiod"] = 9 if intraday else 14

  elif ind == "rsi":
    # Faster RSI on 1/5-min, moderate on 15/30/60, standard otherwise
    if resolution in {"1", "5"}:
      params["timeperiod"] = 5
    elif resolution in {"15", "30", "60"}:
      params["timeperiod"] = 9
    else:
      params["timeperiod"] = 14

  elif ind in {"cci", "cmo", "roc", "rocr", "adx", "adxr", "willr", "mfi",
               "ultosc", "dx", "minusdi", "plusdi", "minusdm", "plusdm", "atr",
               "natr", "mom"}:
    params["timeperiod"] = 10 if intraday else 14

  elif ind in {"macd", "macdext"}:
    # Fast MACD for intraday, classic for daily
    if intraday:
      params["fastperiod"] = 3
      params["slowperiod"] = 10
      params["signalperiod"] = 16
    else:
      params["fastperiod"] = 12
      params["slowperiod"] = 26
      params["signalperiod"] = 9

  elif ind == "stoch":
    params["fastkperiod"] = 5
    params["slowkperiod"] = 3
    params["slowdperiod"] = 3

  elif ind == "stochf":
    params["fastkperiod"] = 5
    params["fastdperiod"] = 3

  elif ind == "stochrsi":
    params["timeperiod"] = 14
    params["fastkperiod"] = 5
    params["fastdperiod"] = 3

  elif ind in {"apo", "ppo"}:
    params["fastperiod"] = 12
    params["slowperiod"] = 26

  elif ind == "adosc":
    params["fastperiod"] = 3
    params["slowperiod"] = 10

  elif ind == "ultosc":
    params["timeperiod1"] = 7
    params["timeperiod2"] = 14
    params["timeperiod3"] = 28

  elif ind == "bbands":
    # Standard Bollinger Bands remain at 20,2 even intraday
    params["timeperiod"] = 20
    params["nbdevup"] = 2
    params["nbdevdn"] = 2

  # Fetch and handle no-data case
  data = _call_finnhub("/indicator", params, session)
  # Truncate all list-valued keys to the last 5 values
  if isinstance(data, dict):
    for key, value in data.items():
      if isinstance(value, list):
        data[key] = value[-5:]
  if data.get("s") == "no_data":
    return None

  _TECHNICAL_CACHE[cache_key] = data
  return data


def fetch_all(symbol: str, resolution: str = "D", lookback_days: int = 2,
    save_path: str | Path = Path("data"),
    session: Optional[requests.Session] = None,
    dry_run: bool = False, ) -> Path:
  """High-level façade to fetch & save all endpoints."""
  global _API_CALL_COUNT
  now = datetime.utcnow().replace(tzinfo=timezone.utc)

  if dry_run:
    logger.info("Dry-run enabled - skipping Finnhub calls for %s", symbol)
    result = {"symbol": symbol.upper(), "resolution": resolution,
      "last_updated_utc": datetime.utcnow().isoformat() + "Z", "candles": {},
      "patterns": [], "support_resistance": {}, "aggregate_indicator": {},
      "technical_indicators": {}, }
    path = Path(save_path)
    path.mkdir(parents=True, exist_ok=True)
    outfile = path / f"{symbol.upper()}_analysis_{resolution}.json"
    with outfile.open("w", encoding="utf-8") as f:
      json.dump(result, f, indent=2)
    logger.info("Saved analysis to %s", outfile)
    return outfile

  # Determine ending timestamp and 60-candle lookback purely in UTC
  end = _get_last_candle_time(resolution, now)
  resolution = resolution.upper()
  if resolution.isdigit():
    minutes_per_candle = int(resolution)
    start = end - timedelta(minutes=minutes_per_candle * 60)
  elif resolution == "D":
    start = end - timedelta(days=60)
  elif resolution == "W":
    start = end - timedelta(weeks=60)
  else:
    raise ValueError(f"Unsupported resolution: {resolution}")
  logger.info("Fetching data for %s at %s resolution", symbol, resolution)
  logger.info("Resolved UTC start: %s end: %s", start.isoformat(), end.isoformat())
  indicators = ["SMA", "EMA", "WMA", "DEMA", "MACD", "MACDEXT", "STOCH",
    "STOCHF", "RSI", "STOCHRSI", "WILLR", "ADX", "ADXR", "APO", "PPO", "MOM",
    "BOP", "CCI", "CMO", "ROC", "ROCR", "AROON", "AROONOSC", "MFI", "TRIX",
    "ULTOSC", "DX", "MINUSDI", "PLUSDI", "MINUSDM", "PLUSDM", "BBANDS",
    "MIDPOINT", "MIDPRICE", "SAR", "TRANGE", "ATR", "NATR", "AD", "ADOSC",
    "OBV", ]

  sess = session or requests.Session()

  candles = get_candles(symbol, resolution, start, end, sess)
  patterns = get_pattern_recognition(symbol, resolution, sess)
  support_resistance = get_support_resistance(symbol, resolution, sess)
  aggregate_indicator = get_aggregate_indicator(symbol, resolution, sess)

  result = {"symbol": symbol.upper(), "resolution": resolution,
    "last_updated_utc": end.isoformat() + "Z",
    "candles": candles.dict() if candles else {},
    "patterns": patterns.dict().get("points", []) if patterns else [],
    "support_resistance": support_resistance.dict() if support_resistance else {},
    "aggregate_indicator": aggregate_indicator.dict() if aggregate_indicator else {}, }
  technical_indicators: dict[str, Any] = {}
  # Weekly resolution does not require the heavy technical indicator fetch
  if resolution != "W":
    for ind in indicators:
      try:
        indicator_response = get_technical_indicator(symbol, ind, resolution,
                                                     start, end, session=sess)
        if indicator_response:
          for key in ("o", "h", "l", "c", "v", "t", "s"):
            indicator_response.pop(key, None)
          technical_indicators[ind] = indicator_response
      except Exception as e:
        logger.warning(f"Failed to fetch technical indicator {ind}: {e}")
  result["technical_indicators"] = technical_indicators

  path = Path(save_path)
  path.mkdir(parents=True, exist_ok=True)
  outfile = path / f"{symbol.upper()}_analysis_{resolution}.json"
  with outfile.open("w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)
  logger.info("Saved analysis to %s", outfile)
  logger.info("Finnhub API calls for %s: %d", symbol.upper(), _API_CALL_COUNT)
  _API_CALL_COUNT = 0
  return outfile

def _get_last_candle_time(resolution: str, now: datetime) -> datetime:
  """Return the timestamp of the last fully completed candle in UTC."""
  if now.tzinfo is None:
    now = now.replace(tzinfo=timezone.utc)
  else:
    now = now.astimezone(timezone.utc)

  resolution = resolution.upper()
  if resolution.isdigit():
    step = int(resolution)
    floored = now.replace(second=0, microsecond=0)
    minute_floor = (floored.minute // step) * step
    last = floored.replace(minute=minute_floor)
    if last >= now:
      last -= timedelta(minutes=step)
    return last
  if resolution == "D":
    day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if day >= now:
      day -= timedelta(days=1)
    return day
  if resolution == "W":
    week_start = now - timedelta(days=now.weekday())
    week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
    if week_start >= now:
      week_start -= timedelta(weeks=1)
    return week_start
  raise ValueError(f"Unsupported resolution: {resolution}")

def main() -> None:
  """CLI entry point."""
  # Example usage:
  q = get_quote("AAPL")
  print(q.c, q.h, q.l, q.o, q.pc, q.t)

  news = get_company_news("AAPL", datetime.utcnow() - timedelta(days=1),
                          datetime.utcnow())
  for item in news:
    print(item)


if __name__ == "__main__":
  main()
