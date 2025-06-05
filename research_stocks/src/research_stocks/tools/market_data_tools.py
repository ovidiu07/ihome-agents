# src/tools/market_data_tools.py
import os, requests, datetime, json
from crewai.tools import BaseTool
from crewai.tools.base_tool import BaseTool

# ------------- DATA HARVESTER TOOLS ----------------------------------- #

class PoliticalNewsTool(BaseTool):
  name : str ="PoliticalNewsTool"
  description : str ="Fetch top political or macro headlines."

  def _run(self, query: str = "politics OR policy", days_back: int = 1):
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
      return {"error": "NEWSAPI_KEY not set"}
    url = (
      "https://newsapi.org/v2/everything"
      f"?q={query}&from={days_back}d&sortBy=publishedAt&pageSize=100"
      f"&apiKey={api_key}"
    )
    return requests.get(url, timeout=20).json()["articles"]


class ETFDataTool(BaseTool):
  name : str ="ETFDataTool"
  description : str ="Return daily metrics for a list of ETF symbols."

  def _run(self, symbols: list[str] = None):
    symbols = symbols or ["SPY", "QQQ", "DIA"]
    key = os.getenv("POLYGON_KEY")
    out = {}
    for sym in symbols:
      url = f"https://api.polygon.io/v2/aggs/ticker/{sym}/prev?apiKey={key}"
      out[sym] = requests.get(url, timeout=10).json()["results"][0]
    return out


class EquityFundamentalsTool(BaseTool):
  name : str ="EquityFundamentalsTool"
  description : str ="Pull latest EPS, revenue, guidance for a ticker."

  def _run(self, ticker: str):
    key = os.getenv("ALPHAVANTAGE_KEY")
    url = (
      "https://www.alphavantage.co/query"
      f"?function=EARNINGS&symbol={ticker}&apikey={key}"
    )
    data = requests.get(url, timeout=15).json()
    return data.get("annualEarnings", [])[:1]  # most recent year


class GlobalEventsTool(BaseTool):
  name : str ="GlobalEventsTool"
  description : str ="Fetch calendar of central-bank decisions & geo events."

  def _run(self, days_ahead: int = 3):
    # Stub output; replace with real calendar API
    return [
      {"event": "ECB rate decision", "time": "2025-06-06 15:15 CET",
       "likely_impact": "High", "region": "EU"}
    ]


class SentimentScanTool(BaseTool):
  name : str ="SentimentScanTool"
  description : str ="Return VIX, put/call ratio, Twitter + Reddit sentiment."

  def _run(self):
    return {
      "vix": 17.8,
      "put_call_ratio": 0.92,
      "twitter_score": +0.14,
      "reddit_score": -0.05,
    }

# ------------- VALUATION TOOLS ---------------------------------------- #

class FundamentalMathTool(BaseTool):
  name : str ="FundamentalMathTool"
  description : str ="Compute valuation ratios from raw fundamentals."

  def _run(self, eps: float, price: float, ebitda: float, shares: float):
    pe = price / eps if eps else None
    ev_ebitda = (price * shares) / ebitda if ebitda else None
    return {"pe_ratio": pe, "ev_ebitda": ev_ebitda}


class HistoricalFinancialsTool(BaseTool):
  name : str ="HistoricalFinancialsTool"
  description : str ="Return 5-year history of EPS & revenue for a ticker."

  def _run(self, ticker: str):
    # Replace with real S3 / DB lookup
    return {"2020": {"eps": 2.1}, "2021": {"eps": 2.3}}

# ------------- TECHNICAL TOOLS ---------------------------------------- #

class MarketPriceTool(BaseTool):
  name : str ="MarketPriceTool"
  description : str ="Fetch OHLC price history."

  def _run(self, ticker: str, days: int = 30):
    key = os.getenv("POLYGON_KEY")
    end = datetime.date.today()
    start = end - datetime.timedelta(days=days)
    url = (
      f"https://api.polygon.io/v2/aggs/ticker/{ticker}"
      f"/range/1/day/{start}/{end}?apiKey={key}&adjusted=true&sort=asc"
    )
    return requests.get(url, timeout=15).json()["results"]


class TALibTool(BaseTool):
  name : str ="TALibTool"
  description : str ="Run TA-Lib indicators on price series."

  def _run(self, close_prices: list[float]):
    import numpy as np, talib
    rsi = float(talib.RSI(np.array(close_prices), timeperiod=14)[-1])
    macd, macdsig, _ = talib.MACD(np.array(close_prices))
    return {"rsi_14": rsi, "macd": macd[-1], "macd_signal": macdsig[-1]}

# ------------- REPORTING TOOLS ---------------------------------------- #

class MarkdownFormatterTool(BaseTool):
  name : str ="MarkdownFormatterTool"
  description : str ="Format a dict of sections into markdown."

  def _run(self, draft_dict: dict):
    md = []
    for h, body in draft_dict.items():
      md.append(f"## {h}\n{body}\n")
    return "\n".join(md)


class SlackPosterTool(BaseTool):
  name : str ="SlackPosterTool"
  description : str ="Post a file or message to a Slack channel."

  def _run(self, channel: str, text: str):
    token = os.getenv("SLACK_BOT_TOKEN")
    if not token:
      return {"error": "SLACK_BOT_TOKEN not set"}
    url = "https://slack.com/api/chat.postMessage"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"channel": channel, "text": text}
    return requests.post(url, headers=headers, json=payload).json()


class GrammarCheckTool(BaseTool):
  name : str ="GrammarCheckTool"
  description : str ="Run a quick grammar pass using LLM."

  def _run(self, text: str):
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model="openai/gpt-4",
        messages=[{"role": "system", "content": "fix grammar"}, {"role": "user", "content": text}],
        temperature=0.2,
        max_tokens= len(text)//3
    )
    return resp.choices[0].message.content