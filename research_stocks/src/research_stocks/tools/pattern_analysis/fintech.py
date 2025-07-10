"""Finnhub pattern analysis wrapper."""

from __future__ import annotations

import json
import logging
import os
import requests
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, RootModel
from typing import Any, Dict, Optional, List

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

BASE_URL = "https://finnhub.io/api/v1"
BACKOFF_FACTOR = 1.5
MAX_RETRIES = 3


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


def get_quote(
    symbol: str,
    session: Optional[requests.Session] = None,
) -> QuoteResponse:
  """
  Return a real-time quote for `symbol`.
  https://finnhub.io/docs/api/quote
  """
  data = _call_finnhub("/quote", {"symbol": symbol.upper()}, session)
  return QuoteResponse.model_validate(data)


class CompanyNewsItem(BaseModel):
  """Single company news item from Finnhub API."""
  category: str                            # Category, e.g., 'company'
  datetime: int | str                      # UNIX timestamp or ISO8601 UTC string
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


def get_company_news(
    symbol: str,
    start: datetime,
    end: datetime,
    session: Optional[requests.Session] = None,
) -> List[CompanyNewsItem]:
  """
  Return company news for `symbol` from `start` to `end`.
  https://finnhub.io/docs/api/company-news
  """
  params = {
    "symbol": symbol.upper(),
    "from": start.strftime("%Y-%m-%d"),
    "to":   end.strftime("%Y-%m-%d"),
  }
  data = _call_finnhub("/company-news", params, session)
  resp = CompanyNewsResponse.model_validate(data)
  return resp.root


def _get_token() -> str:
  """Return API token from environment or raise."""
  load_dotenv()
  token = os.getenv("FINNHUB_TOKEN")
  if not token:
    raise ValueError("FINNHUB_TOKEN not set in environment")
  return token


def _call_finnhub(
    path: str,
    params: Dict[str, Any],
    session: Optional[requests.Session] = None,
) -> Any:
  """Perform a GET request with retries and backoff."""
  token = _get_token()
  params = dict(params)
  params["token"] = token
  url = f"{BASE_URL}{path}"
  sess = session or requests.Session()

  delay = 1.0
  for attempt in range(1, MAX_RETRIES + 1):
    try:
      resp = sess.get(url, params=params, timeout=10)
      if resp.status_code >= 400:
        raise requests.HTTPError(f"{resp.status_code} error: {resp.text}", response=resp)
      data = resp.json()
      if not data:
        raise ValueError("Empty response")
      time.sleep(1)
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
  t: list[str]   = Field(..., description="ISO time stamps")
  s: str         = Field(..., description="Response status")

  @field_validator("t", mode="before")
  def _to_iso(cls, v: list[int]) -> list[str]:
    return [datetime.utcfromtimestamp(ts).isoformat() + "Z" for ts in v]

  class Config:
    extra = "allow"


class PatternRecognitionResponse(BaseModel):
  """Detected chart patterns."""
  symbol: str
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
  symbol: str
  technicalAnalysis: dict[str, Any]

  class Config:
    extra = "allow"


class TechnicalIndicatorResponse(BaseModel):
  """Custom technical indicator data."""
  symbol: str
  indicator: str
  data: dict[str, Any]

  class Config:
    extra = "allow"


def get_candles(
    symbol: str,
    resolution: str,
    start: datetime,
    end: datetime,
    session: Optional[requests.Session] = None,
) -> CandleResponse:
  """Return OHLCV candles for ``symbol`` between ``start`` and ``end``."""
  params = {
    "symbol":     symbol.upper(),
    "resolution": resolution,
    "from":       int(start.timestamp()),
    "to":         int(end.timestamp()),
  }
  data = _call_finnhub("/stock/candle", params, session)
  return CandleResponse.model_validate(data)


def get_pattern_recognition(
    symbol: str,
    session: Optional[requests.Session] = None,
) -> PatternRecognitionResponse:
  """Return detected classical patterns for ``symbol``."""
  data = _call_finnhub("/scan/pattern", {"symbol": symbol.upper()}, session)
  return PatternRecognitionResponse.model_validate(data)


def get_support_resistance(
    symbol: str,
    session: Optional[requests.Session] = None,
) -> SupportResistanceResponse:
  """Return support and resistance levels for ``symbol``."""
  data = _call_finnhub("/scan/support-resistance", {"symbol": symbol.upper()}, session)
  return SupportResistanceResponse.model_validate(data)


def get_aggregate_indicator(
    symbol: str,
    session: Optional[requests.Session] = None,
) -> AggregateIndicatorResponse:
  """Return Finnhub's aggregate technical indicator for ``symbol``."""
  data = _call_finnhub("/scan/technical-indicator", {"symbol": symbol.upper()}, session)
  return AggregateIndicatorResponse.model_validate(data)


def get_technical_indicator(
    symbol: str,
    indicator: str = "rsi",
    resolution: str = "D",
    start: datetime | None = None,
    end: datetime | None = None,
    timeperiod: int = 14,
    session: Optional[requests.Session] = None,
) -> TechnicalIndicatorResponse:
  """Return custom technical indicator data for ``symbol``."""
  if start is None:
    start = datetime.utcnow() - timedelta(days=365)
  if end is None:
    end = datetime.utcnow()
  params = {
    "symbol":     symbol.upper(),
    "indicator":  indicator,
    "resolution": resolution,
    "from":       int(start.timestamp()),
    "to":         int(end.timestamp()),
    "timeperiod": timeperiod,
  }
  data = _call_finnhub("/indicator", params, session)
  return TechnicalIndicatorResponse.model_validate(data)


def fetch_all(
    symbol: str,
    resolution: str = "D",
    lookback_days: int = 365,
    save_path: str | Path = Path("data"),
    session: Optional[requests.Session] = None,
) -> Path:
  """High-level façade to fetch & save all endpoints."""
  end = datetime.utcnow()
  start = end - timedelta(days=lookback_days)

  candles = get_candles(symbol, resolution, start, end, session)
  patterns = get_pattern_recognition(symbol, session)
  support_resistance = get_support_resistance(symbol, session)
  aggregate_indicator = get_aggregate_indicator(symbol, session)
  technical_indicators = get_technical_indicator(symbol, resolution=resolution,
                                                 start=start, end=end, session=session)

  result = {
    "symbol":                symbol.upper(),
    "last_updated_utc":      end.isoformat() + "Z",
    "candles":               candles.dict(),
    "patterns":              patterns.dict().get("points", []),
    "support_resistance":    support_resistance.dict(),
    "aggregate_indicator":   aggregate_indicator.dict(),
    "technical_indicators":  technical_indicators.dict(),
  }

  path = Path(save_path)
  path.mkdir(parents=True, exist_ok=True)
  outfile = path / f"{symbol.upper()}_analysis.json"
  with outfile.open("w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)
  logger.info("Saved analysis to %s", outfile)
  return outfile


def main() -> None:
  """CLI entry point."""
  # Example usage:
  q = get_quote("AAPL")
  print(q.c, q.h, q.l, q.o, q.pc, q.t)

  news = get_company_news("AAPL", datetime.utcnow() - timedelta(days=1), datetime.utcnow())
  for item in news:
    print(item)


if __name__ == "__main__":
  main()