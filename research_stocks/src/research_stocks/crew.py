# import agentops
import yaml
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import WebsiteSearchTool, ScrapeWebsiteTool, TXTSearchTool
from dotenv import load_dotenv
from functools import lru_cache
from pathlib import Path

from tools.market_data_tools import (PoliticalNewsTool, ETFDataTool,
                                     EquityFundamentalsTool, SentimentScanTool,
                                     GlobalEventsTool, MarkdownFormatterTool,
                                     SlackPosterTool, GrammarCheckTool)

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
                 llm=get_appropriate_llm("low"),
                 tools=[PoliticalNewsTool() ])

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
      query_string = (
        f"({symbol_input} financial results OR {symbol_input} quarterly earnings OR {symbol_input} revenue OR "
        f"{symbol_input} profit margins OR {symbol_input} stock movement OR {symbol_input} analyst opinions OR "
        f"{symbol_input} institutional investors OR {symbol_input} sector outlook OR {symbol_input} recent news OR "
        f"{symbol_input} market analysis OR {symbol_input} press releases OR macroeconomic influences like inflation, "
        f"interest rates, or central bank policies affecting {symbol_input})"
      )
      print(">>>> Task input being passed to data_harvester_agent:", {
        "symbol": symbol_input,
        "query": query_string,
        "days_back": 3
      })
      return Task(
        config=self.tasks_yaml().get("harvest_data", {}),
        agent=self.data_harvester_agent(),
        input={
          "symbol": symbol_input,
          "query": query_string,
          "days_back": 3
        },
      )
    except KeyError as e:
      raise RuntimeError(
        f"Task configuration for harvest_data is missing or invalid: {e}")
    except Exception as e:
      raise RuntimeError(f"Error occurred in harvest_data task: {e}")

  # @agent
  # def pattern_scanner_agent(self) -> Agent:
  #   return Agent(config=self.agents_yaml()["pattern_scanner"], verbose=True,
  #                llm=get_appropriate_llm("low"),
  #                tools=[MarketPriceTool(), ForecastSignalTool(),
  #                       PatternRecognitionTool()
  #                       # 🆕 Add technical-sentiment forecast logic
  #                       ])

  @agent
  def report_composer_agent(self) -> Agent:
    return Agent(config=self.agents_yaml()["report_composer"], verbose=True,
                 llm=get_appropriate_llm("low"),
                 tools=[MarkdownFormatterTool(), GrammarCheckTool(),
                        SlackPosterTool(), ])

  # -----------------------------
  # Helper to chunk lists
  # -----------------------------
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


  @crew
  def build_market_brief(self) -> Crew:
      """Creates the Market Briefing Crew based on the provided stock symbol."""
      print(f"Starting Market Briefing Crew for symbol: {self._symbol}...")
      tasks = [self.harvest_data()]

      part1_task = self.compose_report_part1()
      if part1_task:
          tasks.append(part1_task)

      compose_part2 = self.compose_report_part2()
      if compose_part2 is not None:
          tasks.append(compose_part2)

      return Crew(
          agents=[self.data_harvester_agent(), self.report_composer_agent()],
          tasks=tasks,
          process=Process.sequential,
          verbose=True,
      )