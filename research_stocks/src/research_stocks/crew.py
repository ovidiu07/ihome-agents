# import agentops
import json
import math  # ← NEW
import yaml
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import WebsiteSearchTool, ScrapeWebsiteTool, TXTSearchTool
from dotenv import load_dotenv
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote_plus  # ← NEW

from tools.market_data_tools import (PoliticalNewsTool, MarkdownFormatterTool,
                                     GrammarCheckTool)

load_dotenv()

from crewai import LLM

# Use cheaper model for data gathering / valuation to cut costs
cheap_llm = LLM(model="openai/gpt-3.5-turbo", temperature=0.7, max_tokens=4096,
                top_p=0.9, frequency_penalty=0.1, presence_penalty=0.1, seed=42)

# Higher-quality model for analysis tasks but with a lower token limit
analysis_llm = LLM(model="openai/gpt-4", temperature=0.7, max_tokens=1024,
                   top_p=0.9, frequency_penalty=0.1, presence_penalty=0.1,
                   seed=42)

# Full GPT‑4 model reserved for composing the final report
report_llm = LLM(model="openai/gpt-4", temperature=0.8, max_tokens=2048,
                 top_p=0.9, frequency_penalty=0.1, presence_penalty=0.1,
                 seed=42)

# ------------------------------------------------------------------ #
# 100 high-profile tickers → headline aliases                        #
# ------------------------------------------------------------------ #
SYMBOL_ALIASES = {# 1‒10
  "AAPL": ["Apple", "Apple Inc", "Apple Incorporated"],
  "MSFT": ["Microsoft", "Microsoft Corp", "Microsoft Corporation"],
  "GOOGL": ["Alphabet", "Alphabet Inc", "Google", "Google LLC"],
  "AMZN": ["Amazon", "Amazon.com", "Amazon.com Inc"],
  "TSLA": ["Tesla", "Tesla Inc", "Tesla Motors"],
  "NVDA": ["Nvidia", "NVIDIA Corp", "NVIDIA Corporation"],
  "META": ["Meta Platforms", "Meta", "Facebook", "Facebook Inc"],
  "BRK.A": ["Berkshire Hathaway", "Berkshire Hathaway Inc Class A"],
  "BRK.B": ["Berkshire Hathaway", "Berkshire Hathaway Inc Class B"],
  "V": ["Visa", "Visa Inc"],

  # 11‒20
  "MA": ["Mastercard", "MasterCard Inc"],
  "JPM": ["JPMorgan", "JPMorgan Chase", "J.P. Morgan"],
  "JNJ": ["Johnson & Johnson", "J&J"],
  "WMT": ["Walmart", "Wal-Mart", "Wal-Mart Stores"],
  "UNH": ["UnitedHealth", "UnitedHealth Group"],
  "PG": ["Procter & Gamble", "P&G"], "HD": ["Home Depot", "The Home Depot"],
  "DIS": ["Disney", "The Walt Disney Company"],
  "BAC": ["Bank of America", "BofA"],
  "KO": ["Coca-Cola", "The Coca-Cola Company"],

  # 21‒30
  "PEP": ["PepsiCo", "Pepsi"], "PFE": ["Pfizer", "Pfizer Inc"],
  "MCD": ["McDonald's", "McDonalds Corp"],
  "VZ": ["Verizon", "Verizon Communications"],
  "CSCO": ["Cisco", "Cisco Systems"],
  "CMCSA": ["Comcast", "Comcast Corporation"], "ADBE": ["Adobe", "Adobe Inc"],
  "INTC": ["Intel", "Intel Corporation"],
  "CRM": ["Salesforce", "Salesforce.com"],
  "PYPL": ["PayPal", "PayPal Holdings"],

  # 31‒40
  "NKE": ["Nike", "Nike Inc"], "ORCL": ["Oracle", "Oracle Corporation"],
  "T": ["AT&T", "AT and T"], "ABT": ["Abbott Laboratories", "Abbott"],
  "COST": ["Costco", "Costco Wholesale"], "XOM": ["Exxon Mobil", "ExxonMobil"],
  "CVX": ["Chevron", "Chevron Corp"],
  "LLY": ["Eli Lilly", "Eli Lilly and Company"],
  "MRK": ["Merck", "Merck & Co."], "ABBV": ["AbbVie", "AbbVie Inc"],

  # 41‒50
  "AVGO": ["Broadcom", "Broadcom Inc"], "TXN": ["Texas Instruments", "TI"],
  "AMD": ["AMD", "Advanced Micro Devices"],
  "QCOM": ["Qualcomm", "QUALCOMM Incorporated"],
  "BA": ["Boeing", "The Boeing Company"],
  "CAT": ["Caterpillar", "Caterpillar Inc"],
  "GS": ["Goldman Sachs", "The Goldman Sachs Group"],
  "AXP": ["American Express", "AmEx"],
  "SPGI": ["S&P Global", "Standard & Poor's Global"],
  "BLK": ["BlackRock", "BlackRock Inc"],

  # 51‒60
  "BKNG": ["Booking Holdings", "Booking.com", "Priceline"],
  "NOW": ["ServiceNow", "ServiceNow Inc"],
  "UPS": ["United Parcel Service", "UPS"], "FDX": ["FedEx", "Federal Express"],
  "EA": ["Electronic Arts", "EA"], "UBER": ["Uber", "Uber Technologies"],
  "LYFT": ["Lyft", "Lyft Inc"], "SQ": ["Block", "Square", "Block Inc"],
  "ROKU": ["Roku", "Roku Inc"], "ZM": ["Zoom", "Zoom Video Communications"],

  # 61‒70
  "SHOP": ["Shopify", "Shopify Inc"],
  "TWTR": ["Twitter", "X Corp", "Twitter Inc"],
  "SNAP": ["Snap", "Snapchat", "Snap Inc"], "F": ["Ford", "Ford Motor Company"],
  "GM": ["General Motors", "GM"], "NFLX": ["Netflix", "Netflix Inc"],
  "DAL": ["Delta Air Lines", "Delta Airlines"],
  "AAL": ["American Airlines", "American Airlines Group"],
  "LUV": ["Southwest Airlines", "Southwest Airlines Co"],
  "UAL": ["United Airlines", "United Airlines Holdings"],

  # 71‒80
  "RCL": ["Royal Caribbean", "Royal Caribbean Group"],
  "CCL": ["Carnival", "Carnival Corporation"],
  "MAR": ["Marriott", "Marriott International"],
  "HLT": ["Hilton", "Hilton Worldwide"],
  "SBUX": ["Starbucks", "Starbucks Corporation"],
  "MDLZ": ["Mondelez", "Mondelez International"],
  "MO": ["Altria", "Altria Group"],
  "PM": ["Philip Morris International", "Philip Morris"],
  "DE": ["Deere", "John Deere"],
  "IBM": ["IBM", "International Business Machines"],

  # 81‒90
  "GE": ["General Electric", "GE"], "CSX": ["CSX", "CSX Corporation"],
  "NSC": ["Norfolk Southern", "Norfolk Southern Corp"],
  "UNP": ["Union Pacific", "Union Pacific Railroad"],
  "BDX": ["Becton Dickinson", "BD"],
  "ISRG": ["Intuitive Surgical", "Intuitive Surgical Inc"],
  "GILD": ["Gilead Sciences", "Gilead"], "AMGN": ["Amgen", "Amgen Inc"],
  "VRTX": ["Vertex Pharmaceuticals", "Vertex"],
  "REGN": ["Regeneron", "Regeneron Pharmaceuticals"],

  # 91‒100
  "ADP": ["ADP", "Automatic Data Processing"], "INTU": ["Intuit", "Intuit Inc"],
  "WDAY": ["Workday", "Workday Inc"],
  "PLTR": ["Palantir", "Palantir Technologies"],
  "COIN": ["Coinbase", "Coinbase Global"],
  "TDOC": ["Teladoc", "Teladoc Health"],
  "CRWD": ["CrowdStrike", "CrowdStrike Holdings"],
  "ZS": ["Zscaler", "Zscaler Inc"], "OKTA": ["Okta", "Okta Inc"],
  "PANW": ["Palo Alto Networks", "Palo Alto Networks Inc"], }

FINANCE_TERMS = [
  # ─ Earnings & Guidance ────────────────────────────────────────
  "financial results", "quarterly earnings", "earnings report",
  "earnings call transcript", "guidance", "revenue", "EPS", "operating margin",
  "gross margin", "cash flow",
  # ─ Filings & Disclosures ──────────────────────────────────────
  "10-K", "10-Q", "8-K", "S-1 filing", "SEC investigation",
  # ─ Analyst & Fund-flow Signals ────────────────────────────────
  "price target", "upgrade", "downgrade", "initiated at buy",
  "coverage resumed", "analyst rating", "institutional ownership", "ETF flows",
  # ─ Capital Allocation & Actions ───────────────────────────────
  "dividend increase", "special dividend", "share buyback",
  "secondary offering", "convertible notes", "capital allocation",
  # ─ M&A / Partnerships / IP ────────────────────────────────────
  "merger", "acquisition", "strategic partnership", "joint venture",
  "licensing deal",
  # ─ Macro & Supply Chain ───────────────────────────────────────
  "sector outlook", "macro headwind", "inflation impact",
  "supply chain disruption", "export controls",
  # ─ Product / Technology Drivers (AI-heavy tickers) ───────────
  "GPU launch", "AI chip", "data center demand", "H100", "product roadmap",
  "foundry capacity",
  # ─ Company PR catch-all ───────────────────────────────────────
  "press release"]

BAD_TERMS = ["gaming review", "video game trailer", "job posting",
             "reddit meme", "giveaway"]

MAX_Q_LEN = 480  # keep a safety margin under NewsAPI’s 500 limit
CHUNK_SIZE = 10  # 10 finance terms ≈ 250 chars incl. OR + spaces


def get_appropriate_llm(task_complexity: str) -> LLM:
  if task_complexity == "low":
    return cheap_llm
  if task_complexity == "medium":
    return analysis_llm
  return report_llm


# AGENTOPS_API_KEY = os.getenv("AGENTOPS_API_KEY") or 'cd414e33-e4a2-44ec-a71f-b30360462ee8'
# agentops.init(
#     api_key=AGENTOPS_API_KEY,
#     default_tags=['crewai']
# )
def harvest_data_offline(symbols: list[str], days_back: int = 3) -> list[dict]:
  """
  Hit NewsAPI in chunks so each query string stays < 500 chars.
  Collapses the responses into one de-duplicated list and saves
  them to raw_news.json.
  """
  if not symbols:
    raise ValueError("symbols list is empty")

  symbol = symbols[0]  # NewsAPI can't do multiple tickers well
  tool = PoliticalNewsTool()

  # split FINANCE_TERMS into equal chunks
  n_chunks = math.ceil(len(FINANCE_TERMS) / CHUNK_SIZE)
  articles: list[dict] = []

  for i in range(n_chunks):
    terms = FINANCE_TERMS[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE]
    q = build_news_query(symbol, terms)

    if len(quote_plus(q)) > MAX_Q_LEN:
      # (rare) fallback – cut the chunk in half
      mid = len(terms) // 2
      terms = terms[:mid]
      q = build_news_query(symbol, terms)

    batch = tool.run(symbol=symbol, query=q, days_back=days_back)
    articles.extend(batch)

  # ------------------------------------------------------------------ #
  #  De-dupe by URL so we don't keep the same story from chunk 1 & 3   #
  # ------------------------------------------------------------------ #
  seen = set()
  deduped = []
  for art in articles:
    url = art.get("url")
    if url and url not in seen:
      deduped.append(art)
      seen.add(url)

  Path("raw_news.json").write_text(json.dumps(deduped, indent=2),
                                   encoding="utf-8")
  print(f"✅ Saved {len(deduped)} unique articles to raw_news.json")
  return deduped


def _or(items):
  return " OR ".join(f'"{t}"' for t in items)


def build_news_query(symbol: str, terms: list[str]) -> str:
  """
    Boolean string compatible with NewsAPI's /v2/everything endpoint.
    Ensures the symbol (or one alias) appears in the searchable fields.
    """
  aliases = SYMBOL_ALIASES.get(symbol.upper(), [])
  # e.g. '"NVDA" OR "Nvidia" OR "Nvidia Corp"'
  company_block = _or([symbol.upper(), *aliases])

  finance_block = _or(terms)
  not_block = _or(BAD_TERMS)

  # Require the company_block to be in title OR description
  return (f'({company_block}) AND ({finance_block}) '
          f'AND NOT ({not_block})')


@CrewBase
class StockAnalysisCrew:
  agents_config = 'config/agents.yaml'
  tasks_config = 'config/tasks.yaml'

  @lru_cache(maxsize=1)
  def agents_yaml(self) -> dict:
    if isinstance(self.agents_config, dict):  # Avoid re-parsing
      return self.agents_config
    with open(Path(self.agents_config), "r") as f:
      return yaml.safe_load(f)

  @lru_cache(maxsize=1)
  def tasks_yaml(self) -> dict:
    if isinstance(self.tasks_config, dict):  # Avoid re-parsing
      return self.tasks_config
    with open(Path(self.tasks_config), "r") as f:
      return yaml.safe_load(f)

  # # ------------------------------------------------------------------ #
  # # Helper accessors for the ETF and equity watch‑lists declared in    #
  # # config/agents.yaml under data_harvester.etf_watchlist / equity_…   #
  # # ------------------------------------------------------------------ #
  # @lru_cache(maxsize=1)
  # def etf_watchlist(self) -> list[str]:
  #   return self.agents_yaml()["data_harvester"]["inputs"]["etf_watchlist"]
  #
  # @lru_cache(maxsize=1)
  # def equity_watchlist(self) -> list[str]:
  #   return self.agents_yaml()["data_harvester"]["inputs"]["equity_watchlist"]

  @agent
  def data_harvester_agent(self) -> Agent:
    return Agent(config=self.agents_yaml()["data_harvester"], verbose=True,
                 llm=get_appropriate_llm("low"), tools=[PoliticalNewsTool()])

  @agent
  def report_composer_agent(self) -> Agent:
    return Agent(config=self.agents_yaml()["report_composer"], verbose=True,
                 llm=get_appropriate_llm("low"),
                 tools=[MarkdownFormatterTool(), GrammarCheckTool()])

  @agent
  def forecast_enhancer_agent(self) -> Agent:
    """Agent that uses an LLM to refine the forecast based on merged JSON."""
    return Agent(config=self.agents_yaml()["forecast_enhancer"], verbose=True,
                 llm=get_appropriate_llm("medium"), )

  @task
  def harvest_data(self) -> Task:
    """
    Executes the task to harvest data, handling potential errors in input and outputs.
    """
    try:
      symbol_list = getattr(self, '_symbol', [])
      if not symbol_list or not isinstance(symbol_list, list):
        raise ValueError("Symbols are invalid or not provided.")

      # Ensure the symbols are correctly formatted as a comma-separated string
      symbol_input = ", ".join(symbol_list)
      query_string = (f'("{symbol_input}") AND ('
                      '"financial results" OR "quarterly earnings" OR revenue OR '
                      '"profit margin" OR "stock movement" OR analyst OR '
                      '"institutional investor" OR "sector outlook" OR "press release"'
                      ')')
      print(">>>> Task input being passed to data_harvester_agent:",
            {"symbol": symbol_input, "query": query_string, "days_back": 3})
      return Task(config=self.tasks_yaml().get("harvest_data", {}),
                  agent=self.data_harvester_agent(),
                  inputs={"symbol": symbol_input, "query": query_string,
                          "days_back": 3}, )
    except KeyError as e:
      raise RuntimeError(
          f"Task configuration for harvest_data is missing or invalid: {e}")
    except Exception as e:
      raise RuntimeError(f"Error occurred in harvest_data task: {e}")

  @task
  def enhance_forecast(self) -> Task | None:
    """Read merged JSON for the first symbol and produce an enhanced forecast."""
    symbols = getattr(self, "_symbol", [])
    if isinstance(symbols, list) and symbols:
      symbol = symbols[0]
    elif isinstance(symbols, str):
      symbol = symbols
    else:
      print("[Warning] enhance_forecast: No symbol available.")
      return None
    merged_path = Path("output") / f"pattern_analysis_results_{symbol}.json"
    if not merged_path.exists():
      print("[Warning] merged JSON not found – skipping enhancer.")
      return None

    merged_text = merged_path.read_text(encoding="utf-8")
    return Task(config=self.tasks_yaml().get("enhance_forecast", {}),
                agent=self.forecast_enhancer_agent(),
                inputs={"symbol": symbol, "merged_json": merged_text})

  def _chunk(self, items: list[str], size: int) -> list[list[str]]:
    """Split a list into fixed-size chunks while preserving order."""
    return [items[i:i + size] for i in range(0, len(items), size)]

  @task
  def compose_report_part1(self) -> Task | None:
    """
    Generate report sections for the first part of symbols with added error handling.
    """
    try:
      symbols = getattr(self, '_symbol', [])

      # Validate symbols
      if not symbols or not isinstance(symbols, list):
        print("[Error] No tickers available or invalid data format.")
        return None

      # Get the first chunk of symbols
      symbol_chunk = self._chunk(symbols, 10)[0] if len(symbols) > 0 else []
      if not symbol_chunk:
        print("[Warning] compose_report_part1: No tickers in the first chunk.")
        return None

      return Task(config=self.tasks_yaml().get("compose_report", {}),
                  agent=self.report_composer_agent(),
                  input={"symbol": ", ".join(symbol_chunk)})
    except KeyError as e:
      print(
          f"[Error] Task configuration for compose_report_part1 is missing: {e}")
      return None
    except Exception as e:
      print(f"[Error] Unexpected error in compose_report_part1: {e}")
      return None

  # @task
  def compose_report_part2(self) -> Task | None:
    """Second half of report: Handles remaining symbols not processed in part 1."""
    symbol = getattr(self, '_symbol', [])

    # Skip task if both watchlists don't have enough tickers for a second chunk.
    has_symbol_part2 = len(symbol) > 10
    if not has_symbol_part2:
      print(
          "[Warning] Skipping compose_report_part2: Not enough tickers to split")
      return None

    # Retrieve second chunks (safe with length check).
    symbol_part2 = self._chunk(symbol, 10)[1] if len(symbol) > 10 else []

    # Skip task if both parts are empty.
    if not symbol_part2:
      print("[Warning] compose_report_part2: No tickers in second chunk.")
      return None

    return Task(config=self.tasks_yaml()["compose_report_followup"],
                agent=self.report_composer_agent(),
                output_file="daily_market_brief.md",
                input={"symbol": ", ".join(symbol_part2), }, )

  def merge_news_into_results(self, symbol: str | None = None):
    """
    Copy or merge headlines from raw_news.json into the corresponding
    pattern‑analysis results file. If `symbol` is omitted, the method
    uses the instance’s `_symbol` list.
    """
    # Resolve the symbol to use
    if symbol is None:
      symbols = getattr(self, "_symbol", [])
      if isinstance(symbols, list) and symbols:
        symbol = symbols[0]
      elif isinstance(symbols, str):
        symbol = symbols
      else:
        raise ValueError("Symbol could not be inferred for merge operation.")

    news_path = Path("raw_news.json")
    results_path = Path("output") / f"pattern_analysis_results_{symbol}.json"

    # Read news list
    if not news_path.exists():
      raise FileNotFoundError("raw_news.json not found. Run harvester first.")

    news = json.loads(news_path.read_text(encoding="utf-8"))

    # Merge into results (create or update)
    if results_path.exists():
      results = json.loads(results_path.read_text(encoding="utf-8"))
    else:
      results = {}

    results["news_headlines"] = news
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"✅ Merged {len(news)} headlines into {results_path}")

  @crew
  def build_market_brief(self) -> None:
    """End-to-end flow for one symbol.

    1) Harvest & de-dupe headlines (no LLM cost)
    2) Merge them into the pattern-analysis JSON
    3) Ask GPT-4 to enhance the forecast
    """
    print(f"Starting Market Briefing Crew for symbol: {self._symbol}...")
    harvest_data_offline(self._symbol, days_back=3)

    self.merge_news_into_results()

    tasks: list[Task] = []

    forecast_task = self.enhance_forecast()
    if forecast_task:
      tasks.append(forecast_task)

    if not tasks:
      print("[Error] No runnable tasks – aborting crew.")
      return

    Crew(tasks=tasks, process=Process.sequential).run()
