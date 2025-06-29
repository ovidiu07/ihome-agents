# import agentops
from functools import lru_cache
from pathlib import Path

import yaml
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import WebsiteSearchTool, ScrapeWebsiteTool, TXTSearchTool
from dotenv import load_dotenv

from tools.market_data_tools import (PoliticalNewsTool, ETFDataTool,
                                     EquityFundamentalsTool, SentimentScanTool,
                                     FundamentalMathTool,
                                     HistoricalFinancialsTool, MarketPriceTool,
                                     GlobalEventsTool, MarkdownFormatterTool,
                                     SlackPosterTool, GrammarCheckTool,
                                     ForecastSignalTool, PatternRecognitionTool)
from tools.ohlc_formatter_tool import OHLCFormatterTool
from tools.run_analysis import main as run_pattern_analysis

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
@CrewBase
class StockAnalysisCrew:
  agents_config = 'config/agents.yaml'
  tasks_config = 'config/tasks.yaml'

  def fundamental_analysis_tasks(self) -> list[Task]:
    """Split fundamental analysis into smaller 2x2 batches to avoid LLM token limits."""
    etf_chunks = self._chunk(self.etf_watchlist(), 2)
    equity_chunks = self._chunk(self.equity_watchlist(), 2)

    tasks = []
    num_tasks = max(len(etf_chunks), len(equity_chunks))
    for i in range(num_tasks):
      etfs = etf_chunks[i] if i < len(etf_chunks) else []
      equities = equity_chunks[i] if i < len(equity_chunks) else []

      if not etfs and not equities:
        continue  # Skip empty tasks

      input_data = {"etf_symbols": ", ".join(etfs),
                    "equity_symbols": ", ".join(equities)}

      tasks.append(Task(config=self.tasks_yaml()["fundamental_analysis"],
                        agent=self.valuation_engine_agent(), input=input_data))

    return tasks

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

  # ------------------------------------------------------------------ #
  # Helper accessors for the ETF and equity watch‑lists declared in    #
  # config/agents.yaml under data_harvester.etf_watchlist / equity_…   #
  # ------------------------------------------------------------------ #
  @lru_cache(maxsize=1)
  def etf_watchlist(self) -> list[str]:
    return self.agents_yaml()["data_harvester"]["inputs"]["etf_watchlist"]

  @lru_cache(maxsize=1)
  def equity_watchlist(self) -> list[str]:
    return self.agents_yaml()["data_harvester"]["inputs"]["equity_watchlist"]

  @agent
  def data_harvester_agent(self) -> Agent:
    return Agent(config=self.agents_yaml()["data_harvester"], verbose=True,
                 llm=get_appropriate_llm("low"),
                 tools=[PoliticalNewsTool(),  # Correctly instantiated objects
                        ETFDataTool(), EquityFundamentalsTool(),
                        SentimentScanTool(), GlobalEventsTool(), ])

  @task
  def harvest_data(self) -> Task:
    # pass ticker lists into the task so the prompt variables can expand
    return Task(config=self.tasks_yaml()["harvest_data"],
                agent=self.data_harvester_agent(),
                input={"etf_symbols": ", ".join(self.etf_watchlist()),
                       "equity_symbols": ", ".join(self.equity_watchlist()),
                       "query": (
                         "((interest rates OR inflation OR recession OR central bank OR Fed OR ECB OR bond yields) "
                         "OR (tech stocks OR semiconductor OR AI OR electric vehicles OR oil prices OR energy policy OR healthcare regulation)) "
                         "AND (NVIDIA OR Tesla OR Apple OR Microsoft OR Amazon OR Meta OR Alphabet OR Netflix OR JPMorgan OR Berkshire OR Exxon OR UnitedHealth)"),
                       "days_back": 2}, )

  @agent
  def valuation_engine_agent(self) -> Agent:
    return Agent(config=self.agents_yaml()["valuation_engine"], verbose=True,
                 llm=get_appropriate_llm("low"), tools=[FundamentalMathTool(),
                                                        # Computes P/E, EV/EBITDA, valuation insights
                                                        HistoricalFinancialsTool(),
                                                        # Provides 3-year EPS data
                                                        EquityFundamentalsTool(),
                                                        # Adds current EPS, partial revenue
                                                        ETFDataTool(),
                                                        # Supports price, shares outstanding
                                                        ])

  @task
  def fundamental_analysis(self) -> Task:
    return Task(config=self.tasks_yaml()["fundamental_analysis"],
                agent=self.valuation_engine_agent(),
                input={"etf_symbols": ", ".join(self.etf_watchlist()),
                       "equity_symbols": ", ".join(self.equity_watchlist()), },
                verbose=True, )

  @agent
  def pattern_scanner_agent(self) -> Agent:
    return Agent(config=self.agents_yaml()["pattern_scanner"], verbose=True,
                 llm=get_appropriate_llm("low"),
                 tools=[MarketPriceTool(), ForecastSignalTool(),
                        PatternRecognitionTool()
                        # 🆕 Add technical-sentiment forecast logic
                        ])

  @task
  def technical_analysis(self) -> Task | None:
    print("Preparing technical analysis input...")
    print("Technical Analysis Inputs:")
    print("  ETF Symbols:", self.etf_watchlist())
    print("  Equity Symbols:", self.equity_watchlist())

    watchlist = self.etf_watchlist() + self.equity_watchlist()
    if not watchlist:  # Skip if watchlist is empty.
      print("[Warning] Skipping technical_analysis: No tickers in watchlist.")
      return None

    price_tool = MarketPriceTool()
    formatter = OHLCFormatterTool()
    summaries = []
    symbol_to_close_prices = {}  # 🆕 collect close prices per ticker

    for symbol in watchlist:
      try:
        raw_ohlc = price_tool._run(ticker=symbol, days=60)
        # 🆕 Extract and store close prices
        close_prices = [entry["close"] for entry in raw_ohlc if
                        "close" in entry]
        if len(close_prices) >= 35:
          symbol_to_close_prices[symbol] = close_prices
        else:
          print(f"⚠️ Not enough close prices for {symbol}")
        summary = formatter._run(ohlc_data=raw_ohlc, symbol=symbol, max_rows=5)
        summaries.append(summary)
      except Exception as e:
        print(f"⚠️ Failed to fetch/format OHLC for {symbol}: {e}")
        continue

    combined_summary = "\n\n".join(summaries)
    if not combined_summary:  # Ensure valid input is passed downstream.
      print(
          "[Warning] Skipping technical_analysis: No formatted data available.")
      return None

    print("  Combined summary:", combined_summary)
    return Task(config=self.tasks_yaml()["technical_analysis"],
                agent=self.pattern_scanner_agent(),
                input={"etf_symbols": ", ".join(self.etf_watchlist()),
                       "equity_symbols": ", ".join(self.equity_watchlist()),
                       "formatted_ohlc_data": combined_summary, # Valid default.
                       "close_price_map": symbol_to_close_prices,
                       # 🆕 For ForecastSignalTool
                       }, )

  @agent
  def report_composer_agent(self) -> Agent:
    return Agent(config=self.agents_yaml()["report_composer"], verbose=True,
                 llm=get_appropriate_llm("high"),
                 tools=[MarkdownFormatterTool(), GrammarCheckTool(),
                        SlackPosterTool(),
                        # 🆕 Optional: send final report to Slack
                        ])

  # -----------------------------
  # Helper to chunk lists
  # -----------------------------
  def _chunk(self, items: list[str], size: int) -> list[list[str]]:
    """Split a list into fixed-size chunks while preserving order."""
    return [items[i:i + size] for i in range(0, len(items), size)]

  @task
  def compose_report_part1(self) -> Task | None:
    """First report part: Generate report sections for part 1 of symbols."""
    etfs = self.etf_watchlist()
    equities = self.equity_watchlist()

    # Skip task if both watchlists are empty.
    if not etfs and not equities:
      print("[Warning] Skipping compose_report_part1: No tickers available")
      return None

    # Create chunks only if needed or valid.
    etf_part1 = self._chunk(etfs, 10)[0] if len(etfs) > 0 else []
    equity_part1 = self._chunk(equities, 6)[0] if len(equities) > 0 else []

    # Skip task if both chunks are empty.
    if not etf_part1 and not equity_part1:
      print("[Warning] compose_report_part1: No tickers in first chunk.")
      return None

    return Task(config=self.tasks_yaml()["compose_report"],
                agent=self.report_composer_agent(),
                input={"etf_symbols": ", ".join(etf_part1),
                       "equity_symbols": ", ".join(equity_part1), }, )

  # @task
  def compose_report_part2(self) -> Task | None:
    """Second half of report: Handles remaining symbols not processed in part 1."""
    etfs = self.etf_watchlist()
    equities = self.equity_watchlist()

    # Skip task if both watchlists don't have enough tickers for a second chunk.
    has_etf_part2 = len(etfs) > 10
    has_equity_part2 = len(equities) > 6
    if not has_etf_part2 and not has_equity_part2:
      print(
          "[Warning] Skipping compose_report_part2: Not enough tickers to split")
      return None

    # Retrieve second chunks (safe with length check).
    etf_part2 = self._chunk(etfs, 10)[1] if len(etfs) > 10 else []
    equity_part2 = self._chunk(equities, 6)[1] if len(equities) > 6 else []

    # Skip task if both parts are empty.
    if not etf_part2 and not equity_part2:
      print("[Warning] compose_report_part2: No tickers in second chunk.")
      return None

    return Task(config=self.tasks_yaml()["compose_report_followup"],
                agent=self.report_composer_agent(),
                output_file="daily_market_brief.md",
                input={"etf_symbols": ", ".join(etf_part2),
                       "equity_symbols": ", ".join(equity_part2), }, )

  @crew
  def crew(self) -> Crew:
    """Creates the Market Briefing Crew"""
    # How to run here main function from run_analysis.py ?
    symbol = self.equity_watchlist()[0] if self.equity_watchlist() else "MSFT"
    run_pattern_analysis(
        symbol)  # tasks = [self.harvest_data()]  # tasks.extend(self.fundamental_analysis_tasks())  # tech_task = self.technical_analysis()  # part1_task = self.compose_report_part1()  # if tech_task:  #   tasks.append(tech_task)  # if part1_task:  #   tasks.append(part1_task)  #  # compose_part2 = self.compose_report_part2()  # if compose_part2 is not None:  #   tasks.append(compose_part2)  #  # return Crew(  #     agents=[self.data_harvester_agent(), self.valuation_engine_agent(),  #             self.pattern_scanner_agent(), self.report_composer_agent()],  #     tasks=tasks, process=Process.sequential, verbose=True, )
