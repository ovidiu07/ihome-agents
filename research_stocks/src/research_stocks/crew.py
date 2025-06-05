from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import WebsiteSearchTool, ScrapeWebsiteTool, TXTSearchTool
from dotenv import load_dotenv
from typing import List
from functools import lru_cache
import agentops
import os
from tools.market_data_tools import (PoliticalNewsTool, ETFDataTool,
                                     EquityFundamentalsTool, GlobalEventsTool,
                                     SentimentScanTool, FundamentalMathTool,
                                     HistoricalFinancialsTool, MarketPriceTool,
                                     TALibTool, MarkdownFormatterTool,
                                     SlackPosterTool, GrammarCheckTool)

load_dotenv()

from crewai import LLM

# Use cheaper model for data gathering / valuation to cut costs
cheap_llm = LLM(model="openai/gpt-3.5-turbo",
                temperature=0.7, max_tokens=512, top_p=0.9,
                frequency_penalty=0.1, presence_penalty=0.1, seed=42)

# Higher-quality model for analysis tasks but with a lower token limit
analysis_llm = LLM(model="openai/gpt-4",
                   temperature=0.7, max_tokens=1024, top_p=0.9,
                   frequency_penalty=0.1, presence_penalty=0.1, seed=42)

# Full GPT‑4 model reserved for composing the final report
report_llm = LLM(model="openai/gpt-4",
                 temperature=0.8, max_tokens=2048, top_p=0.9,
                 frequency_penalty=0.1, presence_penalty=0.1, seed=42)

AGENTOPS_API_KEY = os.getenv("AGENTOPS_API_KEY") or 'cd414e33-e4a2-44ec-a71f-b30360462ee8'
agentops.init(
    api_key=AGENTOPS_API_KEY,
    default_tags=['crewai']
)
@CrewBase
class StockAnalysisCrew:
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    # ------------------------------------------------------------------ #
    # Helper accessors for the ETF and equity watch‑lists declared in    #
    # config/agents.yaml under data_harvester.etf_watchlist / equity_…   #
    # ------------------------------------------------------------------ #
    @lru_cache(maxsize=1)
    def etf_watchlist(self) -> list[str]:
        return self.agents_config["data_harvester"]["etf_watchlist"]

    @lru_cache(maxsize=1)
    def equity_watchlist(self) -> list[str]:
        return self.agents_config["data_harvester"]["equity_watchlist"]

    @agent
    def data_harvester_agent(self) -> Agent:
        return Agent(config=self.agents_config['data_harvester'], verbose=True,
                     llm=cheap_llm, tools=[PoliticalNewsTool(), ETFDataTool(),
                                     EquityFundamentalsTool(), GlobalEventsTool(),
                                     SentimentScanTool(), ])

    @task
    def harvest_data(self) -> Task:
        # pass ticker lists into the task so the prompt variables can expand
        return Task(
            config=self.tasks_config["harvest_data"],
            agent=self.data_harvester_agent(),
            input={
                "etf_symbols":  ", ".join(self.etf_watchlist()),
                "equity_symbols": ", ".join(self.equity_watchlist()),
            },
        )

    @agent
    def valuation_engine_agent(self) -> Agent:
        return Agent(config=self.agents_config['valuation_engine'], verbose=True,
                     llm=cheap_llm,
                     tools=[FundamentalMathTool(), HistoricalFinancialsTool(), ])

    @task
    def fundamental_analysis(self) -> Task:
        return Task(
            config=self.tasks_config["fundamental_analysis"],
            agent=self.valuation_engine_agent(),
            input={
                "etf_symbols": ", ".join(self.etf_watchlist()),
                "equity_symbols": ", ".join(self.equity_watchlist()),
            },
        )

    @agent
    def pattern_scanner_agent(self) -> Agent:
        return Agent(config=self.agents_config['pattern_scanner'], verbose=True,
                     llm=analysis_llm,
                     tools=[MarketPriceTool(), TALibTool(), WebsiteSearchTool(), ])

    @task
    def technical_analysis(self) -> Task:
        # provide the same lists; the task template can iterate or pick per symbol
        return Task(
            config=self.tasks_config["technical_analysis"],
            agent=self.pattern_scanner_agent(),
            input={
                "equity_symbols": ", ".join(self.equity_watchlist()),
            },
        )

    @agent
    def report_composer_agent(self) -> Agent:
        return Agent(config=self.agents_config['report_composer'], verbose=True,
                     llm=report_llm, tools=[MarkdownFormatterTool(),
                                     GrammarCheckTool(), ScrapeWebsiteTool(),
                                     WebsiteSearchTool(), ])


    # -----------------------------
    # Helper to chunk lists
    # -----------------------------
    def _chunk(self, items: list[str], size: int) -> list[list[str]]:
        """Split a list into fixed-size chunks while preserving order."""
        return [items[i:i + size] for i in range(0, len(items), size)]

    @task
    def compose_report_part1(self) -> Task:
        """First half of the watch‑lists."""
        etf_chunks = self._chunk(self.etf_watchlist(), 10)
        equity_chunks = self._chunk(self.equity_watchlist(), 6)
        return Task(
            config=self.tasks_config["compose_report"],
            agent=self.report_composer_agent(),
            input={
                "etf_symbols": ", ".join(etf_chunks[0]),
                "equity_symbols": ", ".join(equity_chunks[0]),
            },
        )

    @task
    def compose_report_part2(self) -> Task:
        """Second half – writes the final file, appending part‑1 output."""
        etf_chunks = self._chunk(self.etf_watchlist(), 10)
        equity_chunks = self._chunk(self.equity_watchlist(), 6)
        return Task(
            config=self.tasks_config["compose_report_followup"],
            agent=self.report_composer_agent(),
            # append part‑1 context automatically via CrewAI memory
            output_file="daily_market_brief.md",
            input={
                "etf_symbols": ", ".join(etf_chunks[1] if len(etf_chunks) > 1 else []),
                "equity_symbols": ", ".join(equity_chunks[1] if len(equity_chunks) > 1 else []),
            },
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Market Briefing Crew"""
        return Crew(
            agents=self.agents,
            tasks=[
                self.harvest_data(),
                self.fundamental_analysis(),
                self.technical_analysis(),
                self.compose_report_part1(),
                self.compose_report_part2(),   # writes the file
            ],
            process=Process.sequential,
            verbose=True,
        )
