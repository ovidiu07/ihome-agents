"""Web front-end and scheduler for the ResearchStocks crew."""

from __future__ import annotations

import boto3
import json
import logging
import os
import re
from apscheduler.schedulers.background import BackgroundScheduler
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
from pytz import timezone

from crew import StockAnalysisCrew
from tools.run_analysis import main as run_pattern_analysis
from lambda_functions.s3_gpt_analysis import handler as gpt_handler

# FOR DOCKER
# from .crew import StockAnalysisCrew
# from ..tools.run_analysis import main as run_pattern_analysis

load_dotenv()


def run_analysis_and_crew(symbol: str, is_general_analysis: bool = True) -> str:
  """
  Run the pattern analysis and then the crew for the given symbol.

  Args:
      symbol: The stock symbol to analyze
      is_general_analysis: If True, perform general analysis (fetch all finnhub data).
                          If False, perform intraday analysis (fetch only intraday data).

  Returns:
      The final report
  """
  # First run the pattern analysis to generate the JSON file
  print(f"Running pattern analysis for {symbol}...")
  run_pattern_analysis(symbol)
  # CrewAI-native invocation:
  crew_instance = StockAnalysisCrew()
  crew_instance._symbol = [symbol]  # ✅ Store symbol globally in the instance
  return crew_instance.build_market_brief(is_general_analysis).kickoff()


def generate_fallback_report():
  return "Analysis failed. No data available."


def safe_run(symbol: str, is_general_analysis: bool = True) -> str:
  try:
    return run_analysis_and_crew(symbol, is_general_analysis)
  except Exception as e:
    logging.error(f"Critical error in execution: {e}")
    return generate_fallback_report()


# ─── Globals ──────────────────────────────────────────────────────────────

LOCAL_TZ = timezone("Europe/Bucharest")
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
S3_BUCKET = os.getenv("S3_BUCKET", "devtailor-transactions")

# Store submitted symbols per day in memory
DAILY_SYMBOLS: dict[str, list[str]] = {}

# ─── FastAPI setup ─────────────────────────────────────────────────────────

app = FastAPI()
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


@app.get("/forecast", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
  """Render the symbol submission form."""
  today = datetime.now(LOCAL_TZ).strftime("%Y-%m-%d")
  symbols = DAILY_SYMBOLS.get(today, [])
  return templates.TemplateResponse("index.html",
      {"request": request, "symbols": symbols})


@app.post("/submit")
def submit_symbols(symbols: str = Form(...)) -> RedirectResponse:
  """Store the submitted symbols for today's date."""
  symbol_list = [s.strip().upper() for s in re.split(r"[,\s]+", symbols) if
                 s.strip()]
  today = datetime.now(LOCAL_TZ).strftime("%Y-%m-%d")
  DAILY_SYMBOLS[today] = symbol_list
  return RedirectResponse("/", status_code=303)


# ─── Forecast logic & scheduler ────────────────────────────────────────────

def process_today_symbols(is_general_analysis: bool = True) -> None:
  """
  Run analysis for today's submitted symbols and upload results to S3.
  Processes symbols in parallel using ThreadPoolExecutor.

  Args:
      is_general_analysis: If True, perform general analysis (fetch all finnhub data).
                          If False, perform intraday analysis (fetch only intraday data).
  """
  today = datetime.now(LOCAL_TZ).strftime("%Y-%m-%d")
  symbols = DAILY_SYMBOLS.get(today, [])
  if not symbols:
    print("No symbols submitted for today.")
    return

  run_time = datetime.now(LOCAL_TZ).strftime("%d-%b-%Y-%H-%M")
  s3_client = boto3.client("s3")
  os.makedirs("output", exist_ok=True)

  def process_symbol(sym):
    """Process a single symbol and upload results to S3."""
    try:
      print(f"\nProcessing {sym} ...")
      safe_run(sym, is_general_analysis)
      result_path = Path("output") / f"pattern_analysis_results_{sym}.json"
      if result_path.exists():
        uploaded_key = f"{run_time}/{sym}.json"
        with open(result_path, "rb") as fh:
          s3_client.put_object(
              Bucket=S3_BUCKET,
              Key=uploaded_key,
              Body=fh.read(),
              ContentType="application/json",
          )
        print(f"Uploaded results for {sym} to s3://{S3_BUCKET}/{run_time}/")

        # Invoke GPT analysis handler for the uploaded file
        try:
          event = {
              "Records": [
                  {
                      "s3": {
                          "bucket": {"name": S3_BUCKET},
                          "object": {"key": uploaded_key},
                      }
                  }
              ]
          }
          response = gpt_handler(event, None)
          analysis_key = None
          if isinstance(response, dict):
            body = response.get("body")
            if body:
              try:
                analysis_key = json.loads(body).get("analysis_key")
              except Exception as parse_exc:
                logging.warning(
                    "Failed to parse GPT handler response for %s: %s", sym, parse_exc
                )
          logging.info("GPT analysis stored for %s at %s", sym, analysis_key)
        except Exception as handler_exc:
          logging.warning("GPT handler failed for %s: %s", sym, handler_exc)
      else:
        logging.warning("Result JSON for %s not found", sym)
    except Exception as e:
      logging.error(f"Error processing symbol {sym}: {e}")

  # Process symbols sequentially to reduce complexity and resource usage
  for sym in symbols:
    process_symbol(sym)


def start_scheduler() -> BackgroundScheduler:
  """Configure and start the APScheduler."""
  scheduler = BackgroundScheduler(timezone=LOCAL_TZ)

  # General analysis times (15:00 and 16:00)
  general_analysis_times = [(15, 0), (16, 0), (16, 30)]
  for hour, minute in general_analysis_times:
    scheduler.add_job(process_today_symbols, "cron", day_of_week="mon-fri",
        hour=hour, minute=minute, kwargs={"is_general_analysis": True}, )

  # Intraday analysis times (17:00, 18:30, and 19:30)
  intraday_analysis_times = [(17, 0), (17, 45), (18, 0), (18, 30), (19, 30)]
  for hour, minute in intraday_analysis_times:
    scheduler.add_job(process_today_symbols, "cron", day_of_week="mon-fri",
        hour=hour, minute=minute, kwargs={"is_general_analysis": False}, )

  scheduler.start()
  return scheduler


# To trigger manually the service, go to http://0.0.0.0:8000/
# Then make a GET request to "http://localhost:8000/run-now?is_general=true"
@app.get("/run-now")
def manual_run(is_general: bool = True):
  """Manually trigger analysis for today's submitted symbols."""
  process_today_symbols(is_general_analysis=is_general)
  return {"status": "Triggered",
          "type": "general" if is_general else "intraday"}


@app.on_event("startup")
def on_startup() -> None:
  """Start the background scheduler when the app launches."""
  global SCHEDULER
  SCHEDULER = start_scheduler()


@app.on_event("shutdown")
def on_shutdown() -> None:
  """Shut down the scheduler gracefully."""
  if SCHEDULER:
    SCHEDULER.shutdown()


if __name__ == "__main__":
  import uvicorn

  uvicorn.run(app, host="0.0.0.0", port=8080)
