# src/tools/market_data_tools.py
import datetime
import hashlib
import json
import logging
import os
import requests
import time
from crewai.tools import BaseTool
from crewai.tools.base_tool import BaseTool
from dotenv import load_dotenv
from pathlib import Path

CACHE_DIR = os.getenv("CACHE_DIR")
load_dotenv()
print(">>> Loading tools.market_data_tools  from:", __file__)

def track_token_usage(model: str, prompt_tokens: int, completion_tokens: int):
  logging.info(
      f"Token usage - Model: {model}, Prompt: {prompt_tokens}, Completion: {completion_tokens}")


class RateLimiter:
  def __init__(self, calls_per_second: int = 1):
    self.calls_per_second = calls_per_second
    self.last_call_time = 0.0

  def wait_if_needed(self):
    current = time.time()
    delta = current - self.last_call_time
    if delta < 1 / self.calls_per_second:
      time.sleep(1 / self.calls_per_second - delta)
    self.last_call_time = time.time()


def _request_with_retries(url: str, *, max_retries: int = 3,
    initial_delay: int = 1, rate_limiter: RateLimiter | None = None):
  delay = initial_delay
  for attempt in range(max_retries):
    try:
      if rate_limiter:
        rate_limiter.wait_if_needed()
      resp = requests.get(url, timeout=15)
      resp.raise_for_status()
      return resp.json()
    except Exception as e:
      if attempt == max_retries - 1:
        raise RuntimeError(
          f"Failed to fetch URL after {max_retries} attempts: {e}")
      time.sleep(delay)
      delay *= 2


def _validate_response(data: dict, required_fields: list[str]):
  if not all(field in data for field in required_fields):
    raise ValueError(
        f"Invalid response: missing required fields {required_fields}")
  return data


def _get_cache_path(name: str, key: str, ttl: int | None = None) -> Path | None:
  if not CACHE_DIR:
    return None
  path = Path(CACHE_DIR)
  path.mkdir(parents=True, exist_ok=True)
  cache_file = path / f"{name}_{key}.json"
  if ttl is not None and cache_file.exists():
    age = time.time() - cache_file.stat().st_mtime
    if age > ttl:
      try:
        cache_file.unlink()
      except FileNotFoundError:
        pass
  return cache_file


# ------------- DATA HARVESTER TOOLS ----------------------------------- #
class PoliticalNewsTool(BaseTool):
  name: str = "PoliticalNewsTool"
  description: str = "Fetch top political or macro headlines."

  def _run(self, query: str = "politics OR policy", days_back: int = 1):
    import hashlib, json, os
    from pathlib import Path

    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
      raise ValueError("Missing NEWSAPI_KEY for PoliticalNewsTool")

    url = (f"https://newsapi.org/v2/everything"
           f"?q={query}&from={days_back}d&sortBy=publishedAt&pageSize=100"
           f"&apiKey={api_key}")

    cache_key = hashlib.sha1(url.encode()).hexdigest()[:8]
    cache_path = _get_cache_path(self.name, cache_key, ttl=3600)

    if cache_path and cache_path.exists():
      with open(cache_path) as f:
        return json.load(f)

    try:
      data = _request_with_retries(url, rate_limiter=RateLimiter(1))
      articles = data.get("articles", [])
    except Exception as e:
      raise RuntimeError(f"Failed to fetch political news: {str(e)}")

    if cache_path:
      with open(cache_path, "w") as f:
        json.dump(articles, f)

    return articles


class ETFDataTool(BaseTool):
  name: str = "ETFDataTool"
  description: str = "Return daily metrics for a list of ETF and equity symbols."

  def _run(self, symbols: list[str] = None):
    # 🛠️ Normalize symbols input from LLM string to list
    if isinstance(symbols, str):
      symbols = [s.strip().upper() for s in symbols.split(",") if s.strip()]

    symbols = symbols or [
      "SPY", "NVDA"
    ]

    key = os.getenv("POLYGON_KEY")
    if not key:
      return {"error": "POLYGON_KEY not set"}


    out = {}
    symbols_str = ",".join(symbols)

    # ✅ Correct endpoint for batch data
    url = ("https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers"
           f"?ticker.any_of={symbols_str}&apiKey={key}")

    cache_key = hashlib.sha1(url.encode()).hexdigest()[:8]
    cache_path = _get_cache_path(self.name, cache_key, ttl=86400)

    if cache_path and cache_path.exists():
      with open(cache_path) as f:
        data = json.load(f)
    else:
      data = _request_with_retries(url, rate_limiter=RateLimiter(2))
      if cache_path:
        with open(cache_path, "w") as f:
          json.dump(data, f)

    for ticker_data in data.get("tickers", []):
      symbol = ticker_data.get("ticker")
      out[symbol] = ticker_data

    wanted_fields = [
      "ticker",
      "todaysChangePerc", "todaysChange",
      ("day", ["o", "h", "l", "c", "v"])   # open, high, low, close, vol
    ]

    out = {}
    for td in data.get("tickers", []):
      sym = td["ticker"]
      day = td.get("day", {})
      out[sym] = {
        "open":   day.get("o"),
        "high":   day.get("h"),
        "low":    day.get("l"),
        "close":  day.get("c"),
        "volume": day.get("v"),
        "chg_pct": td.get("todaysChangePerc"),
      }
    print(">>> Compact payload:", out)
    return out            # tiny per-ticker dict ≈ 150 chars


class EquityFundamentalsTool(BaseTool):
  name: str = "EquityFundamentalsTool"
  description: str = "Pull latest EPS, revenue, and guidance for a ticker from Alpha Vantage."

  def _run(self, ticker: str):
    key = os.getenv("ALPHAVANTAGE_KEY")
    if not key:
      return {"error": "ALPHAVANTAGE_KEY not set"}

    url = f"https://www.alphavantage.co/query?function=EARNINGS&symbol={ticker}&apikey={key}"
    cache_key = hashlib.sha1(url.encode()).hexdigest()[:8]
    cache_path = _get_cache_path(self.name, cache_key, ttl=86400)

    if cache_path and cache_path.exists():
      with open(cache_path) as f:
        data = json.load(f)
    else:
      data = _request_with_retries(url, rate_limiter=RateLimiter(2))
      if cache_path:
        with open(cache_path, "w") as f:
          json.dump(data, f)

    # Extract latest available earnings
    latest = data.get("annualEarnings", [])[0] if data.get(
        "annualEarnings") else {}

    # Reformat to simpler structure
    return {"ticker": ticker.upper(),
            "fiscal_year": latest.get("fiscalDateEnding", "N/A")[:4],
            "eps": float(latest.get("reportedEPS", 0.0))}


class EquityValuationDataTool(BaseTool):
  name: str = "EquityValuationDataTool"
  description: str = "Fetch EBITDA and shares outstanding for a given equity."

  def _run(self, ticker: str):
    key = os.getenv("ALPHAVANTAGE_KEY")
    if not key:
      return {"error": "ALPHAVANTAGE_KEY not set"}

    url = (f"https://www.alphavantage.co/query?function=INCOME_STATEMENT"
           f"&symbol={ticker}&apikey={key}")
    cache_key = hashlib.sha1(url.encode()).hexdigest()[:8]
    cache_path = _get_cache_path(self.name, cache_key, ttl=86400)

    if cache_path and cache_path.exists():
      with open(cache_path) as f:
        data = json.load(f)
    else:
      data = _request_with_retries(url, rate_limiter=RateLimiter(1))
      if cache_path:
        with open(cache_path, "w") as f:
          json.dump(data, f)

    if not data.get("annualReports"):
      return {"error": "No income statement data available."}

    latest = data["annualReports"][0]
    return {"ticker": ticker.upper(), "ebitda": float(latest.get("ebitda", 0)),
            "shares": float(latest.get("commonStockSharesOutstanding", 0))}


class GlobalEventsTool(BaseTool):
  name: str = "GlobalEventsTool"
  description: str = "Fetch macro and stock-specific future-moving headlines for your watchlist."

  def _run(self, days_ahead: int = 3):
    import datetime
    import hashlib
    import json
    import os
    from pathlib import Path

    key = os.getenv("NEWSAPI_KEY")
    if not key:
      return {"error": "NEWSAPI_KEY not set"}

    today = datetime.date.today()
    to_date = today + datetime.timedelta(days=days_ahead)
    from_str = today.isoformat()
    to_str = to_date.isoformat()

    symbols = ["SPY", "NVDA"]

    macro_terms = ["FOMC", "Fed", "CPI", "inflation", "central bank",
                   "rate hike"]
    stock_terms = ["earnings", "forecast", "guidance", "delivery", "launch"]

    queries = [f'({" OR ".join(macro_terms)}) AND {symbol}' for symbol in
               symbols] + [f'({" OR ".join(stock_terms)}) AND {symbol}' for
                           symbol in symbols]

    all_articles = []
    md_lines = ["# Global Events Report\n"]

    for query in queries:
      url = (f"https://newsapi.org/v2/everything?q={query}"
             f"&from={from_str}&to={to_str}&sortBy=publishedAt"
             f"&language=en&pageSize=10&apiKey={key}")

      cache_key = hashlib.sha1(url.encode()).hexdigest()[:8]
      cache_path = _get_cache_path(self.name, cache_key, ttl=3600)
      if cache_path and cache_path.exists():
        with open(cache_path) as f:
          articles = json.load(f)
      else:
        articles = _request_with_retries(url, rate_limiter=RateLimiter(1)).get(
            "articles", [])
        if cache_path:
          with open(cache_path, "w") as f:
            json.dump(articles, f)

      if articles:
        md_lines.append(f"\n## Query: `{query}`\n")

      for a in articles:
        item = {"query": query, "headline": a.get("title"),
                "source": a.get("source", {}).get("name"),
                "publishedAt": a.get("publishedAt"), "url": a.get("url")}
        all_articles.append(item)
        md_lines.append(
            f"- [{item['headline']}]({item['url']}) — *{item['source']}*, {item['publishedAt']}")

    # Save to markdown file
    with open("GlobalEventsTool.md", "w") as md_file:
      md_file.write("\n".join(md_lines))

    return all_articles


class SentimentScanTool(BaseTool):
  name: str = "SentimentScanTool"
  description: str = "Return VIX and (placeholder) put/call ratio using Polygon.io API."

  def _run(self):
    import hashlib
    import json
    import os
    from pathlib import Path

    key = os.getenv("POLYGON_KEY")
    if not key:
      return {"error": "POLYGON_KEY not set"}

    rate_limiter = RateLimiter(1)
    out = {}

    # --- Cache setup ---
    vix_url = f"https://api.polygon.io/v2/aggs/ticker/VIX/prev?adjusted=true&apiKey={key}"
    cache_key = hashlib.sha1(vix_url.encode()).hexdigest()[:8]
    cache_path = _get_cache_path(self.name, cache_key, ttl=3600)  # 1 hour cache

    # --- Load from cache if exists ---
    if cache_path and cache_path.exists():
      with open(cache_path) as f:
        return json.load(f)

    # --- Fetch VIX ---
    try:
      vix_data = _request_with_retries(vix_url, rate_limiter=rate_limiter)
      vix_close = vix_data["results"][0]["c"]
      out["vix"] = vix_close
    except Exception as e:
      out["vix"] = f"Error: {str(e)}"

    # --- Save to cache ---
    if cache_path:
      with open(cache_path, "w") as f:
        json.dump(out, f)

    # --- Write Markdown Output ---
    md_path = Path("SentimentScanTool.md")
    with open(md_path, "w") as md:
      md.write("# Sentiment Scan Summary\n")
      md.write(f"- **VIX**: {out['vix']}\n")

    return out


# ------------- VALUATION TOOLS ---------------------------------------- #

class FundamentalMathTool(BaseTool):
  name: str = "FundamentalMathTool"
  description: str = "Compute valuation ratios (P/E and EV/EBITDA) and give a valuation grade from fundamentals."

  def _run(self, eps: float, price: float, ebitda: float, shares: float):
    import os
    from pathlib import Path

    result = {}
    errors = []
    ev = None
    valuation_notes = []

    # --- P/E Ratio ---
    if eps is not None and eps > 0:
      pe = round(price / eps, 2)
      result["pe_ratio"] = pe
      if pe < 15:
        valuation_notes.append("🟢 P/E indicates undervaluation.")
      elif pe <= 25:
        valuation_notes.append("🟡 P/E is fair-valued.")
      else:
        valuation_notes.append("🔴 P/E is high — potential overvaluation.")
    else:
      result["pe_ratio"] = None
      errors.append("EPS must be > 0 for valid P/E")

    # --- EV/EBITDA Ratio ---
    if ebitda is not None and ebitda > 0:
      ev = price * shares
      ev_ebitda = round(ev / ebitda, 2)
      result["ev_ebitda"] = ev_ebitda
      if ev_ebitda < 8:
        valuation_notes.append("🟢 EV/EBITDA indicates good value.")
      elif ev_ebitda <= 12:
        valuation_notes.append("🟡 EV/EBITDA is moderate.")
      else:
        valuation_notes.append("🔴 EV/EBITDA is high — caution.")
    else:
      result["ev_ebitda"] = None
      errors.append("EBITDA must be > 0 for valid EV/EBITDA")

    # --- Markdown report ---
    md_lines = ["# Fundamental Valuation Analysis\n"]
    md_lines.append(f"- **Price**: {price}")
    md_lines.append(f"- **EPS (Earnings per Share)**: {eps}")
    md_lines.append(f"- **EBITDA**: {ebitda}")
    md_lines.append(f"- **Shares Outstanding**: {shares}")
    if ev is not None:
      md_lines.append(f"- **Enterprise Value (EV)**: {round(ev, 2)}")

    md_lines.append(f"\n## Computed Ratios")
    md_lines.append(f"- **P/E Ratio**: {result['pe_ratio']}")
    md_lines.append(f"- **EV/EBITDA**: {result['ev_ebitda']}")

    if valuation_notes:
      md_lines.append("\n## Valuation Insights")
      for note in valuation_notes:
        md_lines.append(f"- {note}")

    if errors:
      md_lines.append("\n## Errors / Warnings")
      for err in errors:
        md_lines.append(f"- ⚠️ {err}")

    with open("FundamentalMathTool.md", "w") as f:
      f.write("\n".join(md_lines))

    return result


class HistoricalFinancialsTool(BaseTool):
  name: str = "HistoricalFinancialsTool"
  description: str = "Return 3-year history of EPS & revenue with YoY growth."

  def _run(self, ticker: str):
    key = os.getenv("ALPHAVANTAGE_KEY")
    if not key:
      return {"error": "ALPHAVANTAGE_KEY not set"}

    # Fetch earnings data (EPS)
    url_eps = f"https://www.alphavantage.co/query?function=EARNINGS&symbol={ticker}&apikey={key}"
    cache_key = hashlib.sha1(url_eps.encode()).hexdigest()[:8]
    cache_path = _get_cache_path(self.name, cache_key, ttl=86400)
    if cache_path and cache_path.exists():
      with open(cache_path) as f:
        earnings_data = json.load(f)
    else:
      earnings_data = _request_with_retries(url_eps,
                                            rate_limiter=RateLimiter(1))
      if cache_path:
        with open(cache_path, "w") as f:
          json.dump(earnings_data, f)

    eps_items = earnings_data.get("annualEarnings", [])[:3]
    result = {}

    for item in eps_items:
      year = item["fiscalDateEnding"][:4]
      result[year] = {"eps": float(item["reportedEPS"])}

    # Add EPS growth
    years = sorted(result.keys(), reverse=True)
    for i in range(len(years) - 1):
      this_eps = result[years[i]]["eps"]
      prev_eps = result[years[i + 1]]["eps"]
      if prev_eps != 0:
        growth = round((this_eps - prev_eps) / prev_eps * 100, 2)
        result[years[i]]["eps_growth_%"] = f"{growth}%"
      else:
        result[years[i]]["eps_growth_%"] = "N/A"

    # Optionally write to markdown
    md_lines = [f"# EPS History for {ticker.upper()}"]
    for year in years:
      data = result[year]
      md_lines.append(
          f"- **{year}**: EPS = {data['eps']}, EPS Growth = {data.get('eps_growth_%', '-')}")
    with open(f"EPS_History_{ticker.upper()}.md", "w") as f:
      f.write("\n".join(md_lines))

    return result


# ------------- TECHNICAL TOOLS ---------------------------------------- #

class MarketPriceTool(BaseTool):
  name: str = "MarketPriceTool"
  description: str = "Fetch historical OHLC price data for a given ticker."

  def _run(self, ticker: str, days: int = 20):
    import hashlib, json, os, datetime
    from pathlib import Path

    if not ticker or not isinstance(ticker, str):
      return {"error": "Invalid or missing ticker symbol"}

    key = os.getenv("POLYGON_KEY")
    if not key:
      return {"error": "POLYGON_KEY not set"}

    end = datetime.date.today()
    start = end - datetime.timedelta(days=days)
    url = (f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/"
           f"{start}/{end}?apiKey={key}&adjusted=true&sort=asc")

    cache_key = hashlib.sha1(url.encode()).hexdigest()[:8]
    cache_path = _get_cache_path(self.name, cache_key, ttl=86400)

    if cache_path and cache_path.exists():
      with open(cache_path) as f:
        raw = json.load(f)
    else:
      raw = _request_with_retries(url, rate_limiter=RateLimiter(2))
      if cache_path:
        with open(cache_path, "w") as f:
          json.dump(raw, f)

    results = raw.get("results")
    if not results:
      return {
        "ticker": ticker,
        "ohlc": [],
        "error": f"No price data returned for {ticker}"
      }

    ohlc_series = []
    for item in results:
      date = datetime.datetime.utcfromtimestamp(item["t"] / 1000).date().isoformat()
      ohlc_series.append({
        "date": date,
        "open": item["o"],
        "high": item["h"],
        "low": item["l"],
        "close": item["c"],
        "volume": item["v"]
      })

    # Write markdown summary with full OHLC
    md_path = Path(f"{ticker.upper()}_MarketPrice.md")
    with open(md_path, "w") as f:
      f.write(f"# Market Price Data for {ticker.upper()}\n")
      f.write(f"Date Range: {start} to {end}\n\n")
      for entry in ohlc_series[-5:]:
        f.write(
            f"{entry['date']}: O={entry['open']} H={entry['high']} "
            f"L={entry['low']} C={entry['close']} | Vol={entry['volume']}\n"
        )

    return ohlc_series


class ForecastSignalTool(BaseTool):
  name: str = "ForecastSignalTool"
  description: str = "Forecast today's price movement and provide signal with technical and sentiment overlay."

  def _run(self, ticker: str, close_prices: list[float], vix: float = None):
    import numpy as np, talib, statistics

    if len(close_prices) < 35:
      return {"error": "Need at least 35 close prices"}

    np_prices = np.array(close_prices)
    last_price = np_prices[-1]
    rsi = float(talib.RSI(np_prices, timeperiod=14)[-1])
    macd, macd_signal, _ = talib.MACD(np_prices)
    last_macd = macd[-1]
    last_macd_signal = macd_signal[-1]

    forecast = {"ticker": ticker.upper(), "current_price": round(last_price, 2),
                "rsi": round(rsi, 2), "macd": round(last_macd, 4),
                "macd_signal": round(last_macd_signal, 4), }

    # Technical signal
    if rsi < 30:
      direction = "up"
      base_advice = "🟢 RSI indicates oversold. BUY signal."
    elif rsi > 70:
      direction = "down"
      base_advice = "🔴 RSI indicates overbought. SELL signal."
    elif last_macd > last_macd_signal:
      direction = "up"
      base_advice = "🟢 MACD crossover bullish. BUY signal."
    else:
      direction = "down"
      base_advice = "🔴 MACD crossover bearish. SELL signal."

    forecast["direction"] = direction
    forecast["base_advice"] = base_advice

    # VIX interpretation
    if vix is not None:
      forecast["vix"] = vix
      if vix > 20:
        sentiment_note = "⚠️ VIX high: Market volatility may weaken signals."
      elif vix < 15:
        sentiment_note = "✅ VIX low: Signals are more reliable."
      else:
        sentiment_note = "⚠️ VIX moderate: Use caution."

      forecast["advice"] = f"{base_advice} {sentiment_note}"
    else:
      forecast["advice"] = base_advice

    # Estimate volatility
    volatility = round(statistics.stdev(close_prices[-10:]) / last_price * 100,
                       2)
    forecast["expected_volatility_%"] = volatility

    if direction == "up":
      forecast["today_high_estimate"] = round(
          last_price * (1 + volatility / 100), 2)
      forecast["today_low_estimate"] = round(last_price * 0.995, 2)
    else:
      forecast["today_high_estimate"] = round(last_price * 1.005, 2)
      forecast["today_low_estimate"] = round(
          last_price * (1 - volatility / 100), 2)

    # Markdown report
    with open(f"Forecast_{ticker.upper()}.md", "w") as f:
      f.write(f"# {ticker.upper()} Forecast Signal\n")
      f.write(f"- **Current Price**: ${forecast['current_price']}\n")
      f.write(f"- **RSI (14)**: {forecast['rsi']}\n")
      f.write(
          f"- **MACD**: {forecast['macd']}, Signal: {forecast['macd_signal']}\n")
      f.write(f"- **Expected Direction**: {forecast['direction'].upper()}\n")
      f.write(f"- **Volatility Estimate**: {volatility}%\n")
      f.write(f"- **Estimated High**: ${forecast['today_high_estimate']}\n")
      f.write(f"- **Estimated Low**: ${forecast['today_low_estimate']}\n")
      if vix is not None:
        f.write(f"- **VIX**: {vix}\n")
      f.write(f"- **Advice**: {forecast['advice']}\n")

    return forecast

  @staticmethod
  def run_batch_forecast(watchlist: list[str]):
    from tools.market_data_tools import (SentimentScanTool, MarketPriceTool,
                                         ForecastSignalTool)

    sentiment_tool = SentimentScanTool()
    price_tool = MarketPriceTool()
    forecast_tool = ForecastSignalTool()

    # Step 1: Get market sentiment (VIX)
    vix_data = sentiment_tool._run()
    vix = vix_data.get("vix") if isinstance(vix_data, dict) else None

    # Step 2: Loop through tickers
    results = {}
    for symbol in watchlist:
      print(f"Processing {symbol}...")

      price_data = price_tool._run(symbol, days=40)
      close_prices = [entry["close"] for entry in price_data if
                      "close" in entry]

      if len(close_prices) < 35:
        print(f"Not enough data for {symbol}. Skipping.")
        continue

      forecast = forecast_tool._run(symbol, close_prices, vix=vix)
      results[symbol] = forecast

    return results


# ------------- REPORTING TOOLS ---------------------------------------- #

class MarkdownFormatterTool(BaseTool):
  name: str = "MarkdownFormatterTool"
  description: str = "Format a dict of sections into markdown."

  def _run(self, draft_dict: dict):
    md = []
    for h, body in draft_dict.items():
      md.append(f"## {h}\n{body}\n")
    return "\n".join(md)


class SlackPosterTool(BaseTool):
  name: str = "SlackPosterTool"
  description: str = "Post a file or message to a Slack channel."

  def _run(self, channel: str, text: str):
    token = os.getenv("SLACK_BOT_TOKEN")
    if not token:
      return {"error": "SLACK_BOT_TOKEN not set"}
    url = "https://slack.com/api/chat.postMessage"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"channel": channel, "text": text}
    return requests.post(url, headers=headers, json=payload).json()


class GrammarCheckTool(BaseTool):
  name: str = "GrammarCheckTool"
  description: str = "Run a quick grammar pass using LLM."

  def _run(self, text: str):
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(model="openai/gpt-3.5-turbo",
                                          messages=[{"role": "system",
                                                     "content": "fix grammar"},
                                                    {"role": "user",
                                                     "content": text}],
                                          temperature=0.2,
                                          max_tokens=len(text) // 3)
    usage = resp.usage
    track_token_usage("openai/gpt-3.5-turbo", usage.prompt_tokens,
                      usage.completion_tokens)
    return resp.choices[0].message.content


def check_api_health() -> dict:
  apis = {"polygon": "https://api.polygon.io/v2/reference/status",
          "alphavantage": "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=5min&apikey=demo",
          "newsapi": "https://newsapi.org/v2/top-headlines?country=us&apiKey=demo", }
  results = {}
  for name, url in apis.items():
    try:
      resp = requests.get(url, timeout=5)
      results[name] = resp.status_code == 200
    except Exception:
      results[name] = False
  return results
