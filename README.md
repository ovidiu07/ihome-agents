# ihome-agents

This repository hosts the **ResearchStocks** crew built with [crewAI](https://crewai.com). The agents gather market data, analyse fundamentals and technicals and produce a daily market brief.

## Quick start

1. Install Python 3.10 or newer.
2. Install [uv](https://docs.astral.sh/uv/) and project dependencies:
   ```bash
   pip install uv
   crewai install
   ```
3. Create a `.env` file with your API keys (`OPENAI_API_KEY`, `NEWSAPI_KEY`, `POLYGON_KEY`, etc.).
4. (Optional) set `CACHE_DIR` to reuse API responses across runs.
5. Run the crew:
   ```bash
   crewai run
   ```

The project now includes token usage monitoring and automatic error recovery. If the
crew fails, a fallback message is posted to Slack `#market-ops`.

See `research_stocks/README.md` for full documentation and customisation options.
