# import agentops
import json
import logging
import math  # ← NEW
import yaml
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import WebsiteSearchTool, ScrapeWebsiteTool, TXTSearchTool
from dotenv import load_dotenv
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote_plus  # ← NEW
from datetime import datetime, timezone
from lambda_functions.s3_gpt_analysis import call_gpt_action_with_json_content
from lambda_functions.s3_gpt_analysis import save_analysis
from tools.pattern_analysis.fintech import fetch_all
from tools.pattern_analysis.forecasting import next_prediction_from_finnhub

# FOR DOCKER
# from research_stocks.tools.pattern_analysis.fintech import fetch_all
# from research_stocks.tools.pattern_analysis.forecasting import next_prediction_from_finnhub
#
# from research_stocks.tools.market_data_tools import (PoliticalNewsTool, MarkdownFormatterTool,
#                                      GrammarCheckTool)

# Load environment variables from .env file
load_dotenv()

# Import LLM after environment variables are loaded to ensure proper configuration
from crewai import LLM

# Use cheaper model for data gathering / valuation to cut costs
# GPT-3.5 is used for tasks that don't require deep analysis or complex reasoning
cheap_llm = LLM(model="openai/gpt-3.5-turbo", temperature=0.7, max_tokens=4096,
                top_p=0.9, frequency_penalty=0.1, presence_penalty=0.1, seed=42)

# Higher-quality model for analysis tasks but with a lower token limit
# GPT-4 with reduced token limit for balanced performance and cost efficiency
analysis_llm = LLM(model="openai/gpt-4", temperature=0.7, max_tokens=1024,
                   top_p=0.9, frequency_penalty=0.1, presence_penalty=0.1,
                   seed=42)

# Full GPT‑4 model reserved for composing the final report
# Higher temperature (0.8) allows for more creative output in the final report
report_llm = LLM(model="openai/gpt-4", temperature=0.8, max_tokens=2048,
                 top_p=0.9, frequency_penalty=0.1, presence_penalty=0.1,
                 seed=42)

# ------------------------------------------------------------------ #
# 100 high-profile tickers → headline aliases                        #
# ------------------------------------------------------------------ #
SYMBOL_ALIASES = {  # 1‒10
  "AAPL": ["AAPL", "Apple Inc", "Apple Incorporated", "Apple"],
  "MSFT": ["MSFT", "Microsoft Corp", "Microsoft Corporation", "Microsoft"],
  "GOOGL": ["GOOGL", "Alphabet Inc", "Alphabet Class A", "Google LLC",
            "Google"],
  "GOOG": ["GOOG", "Alphabet Inc", "Alphabet Class C", "Google LLC", "Google"],
  "AMZN": ["AMZN", "Amazon.com Inc", "Amazon.com", "Amazon"],
  "TSLA": ["TSLA", "Tesla Inc", "Tesla Motors", "Tesla"],
  "NVDA": ["NVDA", "NVIDIA Corporation", "NVIDIA Corp", "Nvidia"],
  "META": ["META", "Meta Platforms Inc", "Meta Platforms", "Facebook Inc",
           "Facebook"],
  "BRK.A": ["BRK.A", "Berkshire Hathaway Inc Class A", "Berkshire Hathaway"],
  "BRK.B": ["BRK.B", "Berkshire Hathaway Inc Class B", "Berkshire Hathaway"],
  "V": ["V", "Visa Inc", "Visa"], "MA": ["MA", "MasterCard Inc", "Mastercard"],
  "JPM": ["JPM", "JPMorgan Chase", "J.P. Morgan", "JPMorgan"],
  "JNJ": ["JNJ", "Johnson & Johnson", "J&J"],
  "WMT": ["WMT", "Walmart", "Wal-Mart Stores", "Wal-Mart"],
  "UNH": ["UNH", "UnitedHealth Group", "UnitedHealth"],
  "PG": ["PG", "Procter & Gamble", "P&G"],
  "HD": ["HD", "Home Depot", "The Home Depot"],
  "DIS": ["DIS", "Disney", "The Walt Disney Company"],
  "BAC": ["BAC", "Bank of America", "BofA"],
  "KO": ["KO", "Coca-Cola", "The Coca-Cola Company"],

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
  """
  Returns the appropriate LLM instance based on the task complexity level.

  This function selects the most suitable language model based on the complexity
  of the task to balance performance and cost efficiency.

  Args:
      task_complexity: A string indicating the complexity level of the task.
                      Valid values are "low", "medium", or any other value (treated as "high").

  Returns:
      LLM: The appropriate LLM instance for the given complexity level.
          - "low" complexity tasks use the cheaper GPT-3.5 model
          - "medium" complexity tasks use GPT-4 with reduced token limit
          - All other values default to the full GPT-4 model with higher token limit
  """
  if task_complexity == "low":
    return cheap_llm  # Use GPT-3.5 for simpler tasks
  if task_complexity == "medium":
    return analysis_llm  # Use GPT-4 with reduced token limit for medium complexity
  return report_llm  # Default to full GPT-4 for high complexity tasks


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
  """
  Joins a list of items with the OR operator for use in search queries.

  Args:
      items: A list of strings to be joined with OR operators.

  Returns:
      str: A string with each item enclosed in quotes and joined with " OR ".
           Example: '"item1" OR "item2" OR "item3"'
  """
  return " OR ".join(f'"{t}"' for t in items)


def build_news_query(symbol: str, terms: list[str]) -> str:
  """
  Constructs a boolean search query string compatible with NewsAPI's /v2/everything endpoint.

  This function creates a complex search query that ensures the stock symbol or one of its
  aliases appears in the searchable fields, along with relevant finance terms, while
  excluding unwanted terms.

  Args:
      symbol: The stock symbol to search for (e.g., "AAPL", "MSFT").
      terms: A list of finance-related terms to include in the search.

  Returns:
      str: A formatted boolean search query string with three components:
           1. Company block: The symbol and its aliases joined with OR
           2. Finance block: The finance terms joined with OR
           3. Not block: Terms to exclude from results

  Example:
      build_news_query("NVDA", ["earnings", "revenue"])
      Returns: '("NVDA" OR "Nvidia" OR "Nvidia Corp") AND ("earnings" OR "revenue") 
               AND NOT ("gaming review" OR "video game trailer" OR ...)'
  """
  # Get company aliases from the SYMBOL_ALIASES dictionary
  aliases = SYMBOL_ALIASES.get(symbol.upper(), [])

  # Create the company block with the symbol and its aliases
  # e.g. '"NVDA" OR "Nvidia" OR "Nvidia Corp"'
  company_block = _or([symbol.upper(), *aliases])

  # Create the finance terms block
  finance_block = _or(terms)

  # Create the block of terms to exclude
  not_block = _or(BAD_TERMS)

  # Combine all blocks into the final query
  # Require the company_block to be in title OR description
  return (f'({company_block}) AND ({finance_block}) '
          f'AND NOT ({not_block})')


@CrewBase
class StockAnalysisCrew:
  """
  A crew-based system for analyzing stock market data and generating reports.

  This class orchestrates a set of AI agents that work together to:
  1. Harvest financial news and market data for specified stock symbols
  2. Analyze patterns and trends in the collected data
  3. Enhance forecasts using LLM-based analysis
  4. Compose comprehensive market reports

  The crew uses configuration from YAML files to define agents and tasks.
  It leverages different LLM models based on task complexity to balance
  performance and cost efficiency.

  Attributes:
      agents_config: Path to the YAML file containing agent configurations
      tasks_config: Path to the YAML file containing task configurations
  """
  agents_config = 'config/agents.yaml'
  tasks_config = 'config/tasks.yaml'

  @lru_cache(maxsize=1)
  def agents_yaml(self) -> dict:
    """
    Loads and caches the agent configuration from YAML.

    This method reads the agent configuration from the YAML file specified in
    agents_config. It uses lru_cache to avoid re-parsing the file on subsequent calls.

    Returns:
        dict: A dictionary containing the agent configurations.
    """
    if isinstance(self.agents_config,
                  dict):  # Avoid re-parsing if already a dict
      return self.agents_config
    with open(Path(self.agents_config), "r") as f:
      return yaml.safe_load(f)  # Parse YAML into Python dictionary

  @lru_cache(maxsize=1)
  def tasks_yaml(self) -> dict:
    """
    Loads and caches the task configuration from YAML.

    This method reads the task configuration from the YAML file specified in
    tasks_config. It uses lru_cache to avoid re-parsing the file on subsequent calls.

    Returns:
        dict: A dictionary containing the task configurations.
    """
    if isinstance(self.tasks_config,
                  dict):  # Avoid re-parsing if already a dict
      return self.tasks_config
    with open(Path(self.tasks_config), "r") as f:
      return yaml.safe_load(f)  # Parse YAML into Python dictionary

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
    """
    Creates an agent responsible for harvesting financial news and market data.

    This agent collects data from various sources including news APIs and financial
    data providers. It uses a lower-cost LLM model (GPT-3.5) since the task is
    primarily data collection rather than complex analysis.

    Returns:
        Agent: A configured Agent instance with the PoliticalNewsTool for data collection.
    """
    return Agent(config=self.agents_yaml()["data_harvester"], verbose=True,
                 llm=get_appropriate_llm("low"))

  @agent
  def report_composer_agent(self) -> Agent:
    """
    Creates an agent responsible for composing comprehensive market reports.

    This agent takes the analyzed data and creates well-formatted, readable reports.
    It uses formatting and grammar checking tools to ensure high-quality output.
    Despite the creative nature of report writing, it uses a lower-cost LLM model
    to balance cost efficiency.

    Returns:
        Agent: A configured Agent instance with MarkdownFormatterTool and GrammarCheckTool.
    """
    return Agent(config=self.agents_yaml()["report_composer"], verbose=True,
                 llm=get_appropriate_llm("low"))

  @agent
  def forecast_enhancer_agent(self) -> Agent:
    """
    Creates an agent responsible for enhancing stock forecasts using LLM analysis.

    This agent takes the pattern analysis results and news data, then uses a more
    powerful LLM model (GPT-4) to refine the forecast with deeper analysis and
    reasoning. The medium complexity setting balances the need for sophisticated
    analysis with cost considerations.

    Returns:
        Agent: A configured Agent instance with a medium-complexity LLM model.
    """
    return Agent(config=self.agents_yaml()["forecast_enhancer"], verbose=True,
                 llm=get_appropriate_llm("medium"), )

  @task
  def harvest_data(self) -> Task:
    """
    Creates a task to harvest financial news and market data for specified stock symbols.

    This method configures a task that uses the data_harvester_agent to collect
    relevant financial news and data. It constructs a query string that combines
    the stock symbols with financial terms to ensure relevant results.

    The method includes comprehensive error handling to catch configuration issues
    and other potential errors during task creation.

    Returns:
        Task: A configured Task instance for data harvesting.

    Raises:
        RuntimeError: If there are issues with task configuration or execution.
        ValueError: If no valid symbols are provided.
    """
    try:
      # Get the list of stock symbols from the instance attribute
      symbol_list = getattr(self, '_symbol', [])
      if not symbol_list or not isinstance(symbol_list, list):
        raise ValueError("Symbols are invalid or not provided.")

      # Ensure the symbols are correctly formatted as a comma-separated string
      symbol_input = ", ".join(symbol_list)

      # Construct a comprehensive query string with financial terms
      query_string = (f'("{symbol_input}") AND ('
                      '"financial results" OR "quarterly earnings" OR revenue OR '
                      '"profit margin" OR "stock movement" OR analyst OR '
                      '"institutional investor" OR "sector outlook" OR "press release"'
                      ')')

      # Log the inputs being passed to the agent for debugging
      print(">>>> Task input being passed to data_harvester_agent:",
            {"symbol": symbol_input, "query": query_string, "days_back": 3})

      # Create and return the task with proper configuration
      return Task(config=self.tasks_yaml().get("harvest_data", {}),
                  agent=self.data_harvester_agent(),
                  inputs={"symbol": symbol_input, "query": query_string,
                          "days_back": 3}, )
    except KeyError as e:
      # Handle missing or invalid task configuration
      raise RuntimeError(
          f"Task configuration for harvest_data is missing or invalid: {e}")
    except Exception as e:
      # Handle any other unexpected errors
      raise RuntimeError(f"Error occurred in harvest_data task: {e}")

  @task
  def enhance_forecast(self) -> Task | None:
    """
    Creates a task to enhance stock forecasts using LLM analysis of merged data.

    This method reads the merged JSON file containing pattern analysis results and
    news data for the first symbol in the list, then configures a task that uses
    the forecast_enhancer_agent to produce a more sophisticated forecast.

    The method includes error handling for cases where no symbol is available or
    the merged JSON file doesn't exist.

    Returns:
        Task: A configured Task instance for forecast enhancement.
        None: If no symbol is available or the merged JSON file doesn't exist.
    """
    # Get the list of stock symbols from the instance attribute
    symbols = getattr(self, "_symbol", [])

    # Extract the first symbol from the list or use the string directly
    if isinstance(symbols, list) and symbols:
      symbol = symbols[0]  # Use the first symbol in the list
    elif isinstance(symbols, str):
      symbol = symbols  # Use the symbol string directly
    else:
      print("[Warning] enhance_forecast: No symbol available.")
      return None

    # Construct the path to the merged JSON file
    merged_path = Path("output") / f"pattern_analysis_results_{symbol}.json"

    # Check if the merged JSON file exists
    if not merged_path.exists():
      print("[Warning] merged JSON not found – skipping enhancer.")
      return None

    # Read the merged JSON file
    merged_text = merged_path.read_text(encoding="utf-8")

    # Create and return the task with proper configuration
    return Task(config=self.tasks_yaml().get("enhance_forecast", {}),
                agent=self.forecast_enhancer_agent(),
                inputs={"symbol": symbol, "merged_json": merged_text})

  def _chunk(self, items: list[str], size: int) -> list[list[str]]:
    """
    Splits a list into fixed-size chunks while preserving the original order.

    This utility method is used to divide a large list of items (such as stock symbols)
    into smaller, more manageable chunks for processing. This is particularly useful
    when generating reports for multiple stocks, as it allows for better organization
    and potentially parallel processing.

    Args:
        items: The list of strings to be chunked.
        size: The maximum size of each chunk.

    Returns:
        A list of lists, where each inner list contains at most 'size' items
        from the original list, in the same order.

    Example:
        _chunk(["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"], 2)
        Returns: [["AAPL", "MSFT"], ["GOOGL", "AMZN"], ["TSLA"]]
    """
    # Use list comprehension to create chunks of the specified size
    return [items[i:i + size] for i in range(0, len(items), size)]

  @task
  def compose_report_part1(self) -> Task | None:
    """
    Creates a task to compose the first part of the market report for a chunk of symbols.

    This method configures a task that uses the report_composer_agent to generate
    a comprehensive market report for the first chunk of stock symbols (up to 10).
    It includes error handling for cases where no symbols are available or the
    configuration is missing.

    The method uses the _chunk utility to divide the symbols into manageable groups,
    and processes only the first chunk in this method. For additional symbols,
    the compose_report_part2 method can be used.

    Returns:
        Task: A configured Task instance for report composition.
        None: If no symbols are available, the first chunk is empty, or there's a configuration error.
    """
    try:
      # Get the list of stock symbols from the instance attribute
      symbols = getattr(self, '_symbol', [])

      # Validate that symbols is a non-empty list
      if not symbols or not isinstance(symbols, list):
        print("[Error] No tickers available or invalid data format.")
        return None

      # Get the first chunk of symbols (up to 10)
      symbol_chunk = self._chunk(symbols, 10)[0] if len(symbols) > 0 else []
      if not symbol_chunk:
        print("[Warning] compose_report_part1: No tickers in the first chunk.")
        return None

      # Create and return the task with proper configuration
      return Task(config=self.tasks_yaml().get("compose_report", {}),
                  agent=self.report_composer_agent(),
                  input={"symbol": ", ".join(symbol_chunk)})
    except KeyError as e:
      # Handle missing task configuration
      print(
          f"[Error] Task configuration for compose_report_part1 is missing: {e}")
      return None
    except Exception as e:
      # Handle any other unexpected errors
      print(f"[Error] Unexpected error in compose_report_part1: {e}")
      return None

  # @task  # ⚠️ This task is currently disabled (decorator commented out)
  def compose_report_part2(self) -> Task | None:
    """
    Creates a task to compose the second part of the market report for remaining symbols.

    This method is designed to handle the second chunk of stock symbols (symbols 11-20)
    that weren't processed in compose_report_part1. It configures a task that uses
    the report_composer_agent to generate a comprehensive market report for these
    additional symbols.

    Note: This method is currently disabled (the @task decorator is commented out),
    which suggests it may not be actively used in the current workflow or is being
    reserved for future use when handling larger symbol lists.

    Returns:
        Task: A configured Task instance for report composition of the second chunk.
        None: If there aren't enough symbols for a second chunk or the second chunk is empty.
    """
    # Get the list of stock symbols from the instance attribute
    symbol = getattr(self, '_symbol', [])

    # Skip task if there aren't enough tickers for a second chunk (need more than 10)
    has_symbol_part2 = len(symbol) > 10
    if not has_symbol_part2:
      print(
          "[Warning] Skipping compose_report_part2: Not enough tickers to split")
      return None

    # Retrieve the second chunk of symbols (symbols 11-20)
    symbol_part2 = self._chunk(symbol, 10)[1] if len(symbol) > 10 else []

    # Skip task if the second chunk is empty
    if not symbol_part2:
      print("[Warning] compose_report_part2: No tickers in second chunk.")
      return None

    # Create and return the task with proper configuration
    return Task(config=self.tasks_yaml()["compose_report_followup"],
                agent=self.report_composer_agent(),
                output_file="daily_market_brief.md",
                input={"symbol": ", ".join(symbol_part2), }, )

  def merge_news_into_results(self, symbol: str | None = None):
    """
    Merges news headlines from raw_news.json into the pattern analysis results file.

    This method takes news headlines collected by the data harvester and merges them
    into the corresponding pattern analysis results file for a specific stock symbol.
    This combined data is then used by the forecast enhancer to produce more accurate
    and context-aware forecasts.

    The method handles both creating a new results file if one doesn't exist and
    updating an existing file with the news headlines.

    Args:
        symbol: The stock symbol to merge news for. If None, the method attempts
               to use the first symbol from the instance's _symbol attribute.

    Raises:
        ValueError: If no symbol is provided and none can be inferred.
        FileNotFoundError: If the raw_news.json file doesn't exist.
    """
    # Resolve the symbol to use - either use the provided symbol or infer from instance
    if symbol is None:
      symbols = getattr(self, "_symbol", [])
      if isinstance(symbols, list) and symbols:
        symbol = symbols[0]  # Use the first symbol in the list
      elif isinstance(symbols, str):
        symbol = symbols  # Use the symbol string directly
      else:
        raise ValueError("Symbol could not be inferred for merge operation.")

    # Define paths to the news file and results file
    news_path = Path("raw_news.json")
    results_path = Path("output") / f"pattern_analysis_results_{symbol}.json"

    # Check if the news file exists
    if not news_path.exists():
      raise FileNotFoundError("raw_news.json not found. Run harvester first.")

    # Read the news data from the JSON file
    news = json.loads(news_path.read_text(encoding="utf-8"))

    # Either update existing results or create a new results object
    if results_path.exists():
      # Update existing results file
      results = json.loads(results_path.read_text(encoding="utf-8"))
    else:
      # Create new results object if no file exists
      results = {}

    # Add the news headlines to the results
    # results["news_headlines"] = news

    # Ensure the output directory exists
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # Write the updated results back to the file
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Log the successful merge
    print(f"✅ Merged {len(news)} headlines into {results_path}")

  @crew
  def build_market_brief(
      self,
      is_general_analysis: bool = True,
      previous_report: str | None = None,
  ) -> None:
    """
    Executes the end-to-end workflow for generating a market brief for a stock symbol.

    This method orchestrates the complete process of generating a market brief:
    1. Harvests and de-duplicates financial news headlines (without using LLM)
    2. Merges the headlines into the pattern analysis results JSON file
    3. Uses GPT-4 to enhance the forecast based on the combined data

    The method uses a sequential process to ensure that each step is completed
    before the next one begins. It includes error handling to abort the process
    if no runnable tasks are available.

    This is the main entry point for the stock analysis workflow and should be
    called with the target symbol(s) set in the _symbol attribute.

    Args:
        is_general_analysis: If True, perform general analysis (fetch all finnhub data).
                            If False, perform intraday analysis (fetch only intraday data).
        previous_report: Optional text of the most recent general analysis report
            to optimise during intraday runs.

    Returns:
        None
    """
    # Log the start of the process
    print(f"Starting Market Briefing Crew for symbol: {self._symbol}...")
    analysis_type = "general" if is_general_analysis else "intraday"
    print(f"Analysis type: {analysis_type}")

    # ── Step 1: Harvest financial news data (cost-efficient, no LLM) ───────
    # Normalize self._symbol (which may be a list) into a single string
    symbols = self._symbol
    if isinstance(symbols, list) and symbols:
      symbol = symbols[0]
    elif isinstance(symbols, str):
      symbol = symbols
    else:
      raise ValueError("No valid symbol provided for market brief")

    # harvest_data_offline expects a list of symbols
    try:
      # Always fetch intraday data
      fintech_one_minute = fetch_all(symbol, resolution="1", lookback_days=1,
                                     save_path="output")
      fintech_five_minutes = fetch_all(symbol, resolution="5", lookback_days=1,
                                       save_path="output")
      fintech_fifteen_minutes = fetch_all(symbol, resolution="15",
                                          lookback_days=1, save_path="output")
      fintech_hourly = fetch_all(symbol, resolution="60", lookback_days=3,
                                 save_path="output")

      # Fetch daily and weekly data only for general analysis
      if is_general_analysis:
        fintech_daily = fetch_all(symbol, resolution="D", lookback_days=14,
                                  save_path="output")
        fintech_weekly = fetch_all(symbol, resolution="W", lookback_days=14,
                                   save_path="output")
      else:
        fintech_daily = None
        fintech_weekly = None

    except Exception as e:
      logging.warning("Failed to fetch fintech data: %s", e)
      fintech_hourly = None
      fintech_one_minute = None
      fintech_five_minutes = None
      fintech_fifteen_minutes = None
      fintech_daily = None
      fintech_weekly = None

    # Process the fetched data
    results_path = Path("output") / f"pattern_analysis_results_{symbol}.json"
    try:
      results = json.loads(results_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
      results = {}

    print(f"Starting to fetch information for symbol: {symbol}...")
    # Load fetched intraday fintech data
    if fintech_one_minute:
      fintech_data_one_minute = json.loads(
        fintech_one_minute.read_text(encoding="utf-8"))
      results["fintech_one_minute"] = fintech_data_one_minute

    if fintech_five_minutes:
      fintech_data_five_minutes = json.loads(
        fintech_five_minutes.read_text(encoding="utf-8"))
      results["fintech_five_minutes"] = fintech_data_five_minutes

    if fintech_fifteen_minutes:
      fintech_data_fifteen_minutes = json.loads(
        fintech_fifteen_minutes.read_text(encoding="utf-8"))
      results["fintech_fifteen_minutes"] = fintech_data_fifteen_minutes

    if fintech_hourly:
      fintech_data_hourly = json.loads(
        fintech_hourly.read_text(encoding="utf-8"))
      results["fintech_hourly"] = fintech_data_hourly

    # Load daily and weekly data only for general analysis
    if is_general_analysis:
      if fintech_daily:
        fintech_data_daily = json.loads(
          fintech_daily.read_text(encoding="utf-8"))
        results["fintech_daily"] = fintech_data_daily

      if fintech_weekly:
        fintech_data_weekly = json.loads(
          fintech_weekly.read_text(encoding="utf-8"))
        results["fintech_weekly"] = fintech_data_weekly

    # Compute immediate forecast from raw Finnhub payload
    try:
      print(f"Writing next prediction for symbol: {symbol}...")
      results["next_prediction_from_finnhub"] = next_prediction_from_finnhub(
        results)
    except Exception as exc:
      logging.warning("Failed to build next_prediction: %s", exc)

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # ── Step 2: Pull company‐specific news via Finnhub and append ────────
    try:
      from tools.pattern_analysis.fintech import get_company_news

      # start = datetime.utcnow() - timedelta(days=1)
      # end = datetime.utcnow()
      # news_items = get_company_news(symbol, start, end)
      print(f"Creating json file for symbol: {symbol}...")
      results_path = Path("output") / f"pattern_analysis_results_{symbol}.json"
      try:
        results = json.loads(results_path.read_text(encoding="utf-8"))
      except FileNotFoundError:
        results = {}

      # Serialize Pydantic news items
      # results["company_news"] = [item.dict() for item in news_items]
      results_path.parent.mkdir(parents=True, exist_ok=True)
      results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
      result = call_gpt_action_with_json_content(
          results,
          symbol,
          is_general_analysis,
          previous_report,
      )
      analysis_text = result
      if analysis_text is None:
        logger.error("GPT Action response missing 'analysis' field: %s", result)
        raise RuntimeError("Missing analysis in GPT Action response")

      # Build analysis object key
      date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M")
      analysis_key = f"analysis/{date_str}-{symbol}.md"

      # Save analysis result to S3
      save_analysis("devtailor-transactions", analysis_key, analysis_text)
    except Exception as exc:
      logging.warning("Failed to append company news: %s", exc)

    # ── Step 3: Merge any remaining harvested headlines ───────────────────
    # self.merge_news_into_results()

    # ── Step 4: Queue up forecast enhancement if available ───────────────
    tasks: list[Task] = []
    forecast_task = self.enhance_forecast()
    if forecast_task:
      tasks.append(forecast_task)

    if not tasks:
      print("[Error] No runnable tasks – aborting crew.")
      return

    # Execute all tasks in sequence
    Crew(tasks=tasks, process=Process.sequential).run()
