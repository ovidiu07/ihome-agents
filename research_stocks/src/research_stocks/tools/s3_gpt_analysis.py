from __future__ import annotations

import boto3
import json
import logging
import logging
import openai
import os
import re
import requests
# Email sending function
import smtplib
import time
from botocore.exceptions import ClientError
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from openai import OpenAI, OpenAIError
from typing import Dict, List

client = OpenAI()

logger = logging.getLogger()
logger.setLevel(logging.INFO)
app = FastAPI()
s3_client = boto3.client("s3")
openai.api_key = os.getenv("OPENAI_API_KEY")


def generate_presigned_url(bucket: str, key: str, expiration: int = 300) -> str:
  """Generate a presigned URL to access the uploaded S3 object."""
  try:
    url = s3_client.generate_presigned_url("get_object",
                                           Params={"Bucket": bucket,
                                                   "Key": key},
                                           ExpiresIn=expiration, )
    logger.info("Generated presigned URL for %s/%s", bucket, key)
    return url
  except ClientError as e:
    logger.error("Failed to generate presigned URL: %s", e)
    raise


def load_grok_parse_json_instructions(block: dict, symbol: str, timeframe: str) -> List[Dict[str, str]]:
  """Build a messages array (system, developer, user) for GROK parse‑JSON.

  System and developer instructions are loaded from S3 (if available). The user
  message is composed from the provided `results` JSON and `filename`.
  Returns a list of chat messages suitable for OpenAI/xAI chat APIs.
  """
  # ── Load SYSTEM instructions ───────────────────────────────────────────────
  try:
    sys_key = "gpt/grok_parse_json_sys_instructions.txt"
    obj = s3_client.get_object(Bucket="devtailor-transactions", Key=sys_key)
    system_text = obj["Body"].read().decode("utf-8")
  except ClientError as e:
    logger.error("Could not fetch GROK SYSTEM instructions from S3: %s", e)
    system_text = (
      "You are a rigorous intraday market analyst. Output ONE JSON object only, "
      "following the provided schema. Do not include prose or code fences.")

  # ── Build USER (runtime) message ───────────────────────────────────────────
  user_text = (
      f"AUTHORITATIVE HEADER:\n"
      f"symbol: {symbol}\n"
      f"timeframe: {timeframe}\n"
      "Rules: Copy the HEADER symbol and timeframe verbatim into the output. "
      "If the block conflicts, prefer the HEADER.\n\n"
      "Analyze the following finnhub timeframe block and return exactly one JSON object per the schema.\n\n"
      "INPUT_BLOCK:\n" + json.dumps(block, ensure_ascii=False)
  )

  messages: List[Dict[str, str]] = [{"role": "system", "content": system_text},
    {"role": "user", "content": user_text}, ]
  return messages

def load_system_persona_tiny(is_general_analysis: bool = True) -> str:
  """Load system instructions for the GPT call from S3."""
  try:
    key = (
      "gpt/general_system_persona_instructions_v1.txt" if is_general_analysis else "gpt/intraday_system_persona_instructions_v1.txt")
    obj = s3_client.get_object(Bucket="devtailor-transactions", Key=key)
    return obj["Body"].read().decode("utf-8")
  except ClientError as e:
    logger.error("Could not fetch instructions from S3: %s", e)
    return "Default fallback instructions here..."


def load_data_contract_and_scaffold(is_general_analysis: bool = True) -> str:
  """Load system instructions for the GPT call from S3."""
  try:
    key = (
      "gpt/general_data_contract_and_scaffold_instructions_v1.txt" if is_general_analysis else "gpt/intraday_data_contract_and_scaffold_instructions_v1.txt")
    obj = s3_client.get_object(Bucket="devtailor-transactions", Key=key)
    return obj["Body"].read().decode("utf-8")
  except ClientError as e:
    logger.error("Could not fetch instructions from S3: %s", e)
    return "Default fallback instructions here..."


# ---------------------------------------------------------------------------
# Simple markdown-based validator ― tighten as needed
# ---------------------------------------------------------------------------
_HEADINGS_RE = re.compile(r"#+\s*SECTION\s*([12])", re.I)

_PAT_5MIN_TABLE = re.compile(r"5[- ]Minute Forecast[\s\S]*?\|\s*UTC\s*\|", re.I)
_PAT_5MIN_JSON = re.compile(r"5[- ]Minute Forecast[\s\S]*?```json", re.I)


def _validates(markdown: str) -> bool:
  """
  Passes if:
    • SECTION 1 and SECTION 2 headings are present (any markdown level).
    • The 5‑Minute Forecast block contains EITHER a markdown table OR a fenced JSON block.
  """
  if not {"1", "2"}.issubset(set(_HEADINGS_RE.findall(markdown))):
    return False

  if not (_PAT_5MIN_TABLE.search(markdown) or _PAT_5MIN_JSON.search(markdown)):
    return False
  return True


# ---------------------------------------------------------------------------
# Core wrapper
# ---------------------------------------------------------------------------
def call_gpt_action_with_json_content(results: dict, filename: str,
    is_general_analysis: bool = True, previous_report: str | None = None,
    max_retries: int = 1, ) -> str:
  SYSTEM_A = load_system_persona_tiny(is_general_analysis)  # tiny & stable
  DEV_B = load_data_contract_and_scaffold(is_general_analysis)  # long & stable
  json_content = json.dumps(results, indent=2)

  # Build the runtime "C" message
  user_parts = []
  if not is_general_analysis and previous_report:
    user_parts.append("previous_daily_report_md:\n" + previous_report)
  user_parts.append(f"intraday_json (file={filename}):\n{json_content}")
  user_parts.append(
    "TASK: Parse anchors and render SECTIONS 1–3 exactly as per the contract. No extra sections.")
  USER_C = "\n\n".join(user_parts)

  # Build messages and client per provider
  if not is_general_analysis:
    # GROK (xAI): merge developer into system; use xAI key + base_url
    merged_system = (SYSTEM_A or "").strip() + "\n\n" + (DEV_B or "").strip()
    messages = [
      {"role": "system", "content": merged_system},
      {"role": "user", "content": USER_C},
    ]
    client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY"),
        base_url="https://api.x.ai/v1",
    )
    kwargs = {"temperature": 0, "seed": 42}
  else:
    # OpenAI for general analysis: keep developer role
    messages = [
      {"role": "system", "content": SYSTEM_A},
      {"role": "developer", "content": DEV_B},
      {"role": "user", "content": USER_C},
    ]
    model_name = os.getenv("OPENAI_MODEL_GENERAL", "o3")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    kwargs = {}
  resp = client.chat.completions.create(model=model_name, messages=messages, **kwargs)
  content = resp.choices[0].message.content

  # Monitor caching: cached token count appears here on supported models
  try:
    usage = resp.usage
    cached = getattr(usage, "prompt_tokens_details", {}).get("cached_tokens", 0)
    logger.info("finish_reason=%s cached_tokens=%s total_prompt=%s",
                resp.choices[0].finish_reason, cached,
                usage.prompt_tokens if usage else None)
  except Exception:
    pass

  send_email_with_analysis(content, filename)
  return content


def send_email_with_analysis(content: str, subject_filename: str):
  sender_email = os.getenv("SENDER_EMAIL") or "contact@ihomeprosolutions.ro"
  receiver_raw = os.getenv(
      "RECEIVER_EMAIL") or "moldovan.ovidiuv@gmail.com, moldovan.iuliae@gmail.com"
  smtp_server = os.getenv("SMTP_SERVER") or "smtppro.zoho.eu"
  smtp_port = int(os.getenv("SMTP_PORT", 587))
  smtp_username = os.getenv("SMTP_USERNAME") or "contact@ihomeprosolutions.ro"
  smtp_password = os.getenv("SMTP_PASSWORD") or "Marley01042022$"

  receiver_list = [email.strip() for email in receiver_raw.split(",") if
                   email.strip()]

  if not sender_email or not receiver_list:
    logger.error("Missing sender or receiver email. Check configuration.")
    return

  msg = MIMEMultipart()
  msg["From"] = sender_email
  msg["To"] = ", ".join(receiver_list)
  msg["Subject"] = f"Check your new analysis report for: {subject_filename}"
  msg.attach(MIMEText(content, "plain"))

  try:
    with smtplib.SMTP(smtp_server, smtp_port) as server:
      server.starttls()
      server.login(smtp_username, smtp_password)
      server.sendmail(sender_email, receiver_list, msg.as_string())
      logger.info("Sent analysis email to: %s", ", ".join(receiver_list))
  except Exception as e:
    logger.error("Failed to send email: %s", e)


def save_analysis(bucket: str, key: str, analysis: str):
  """Save analysis text to S3 under the specified key."""
  try:
    s3_client.put_object(Bucket=bucket, Key=key, Body=analysis.encode("utf-8"))
    logger.info("Saved analysis to %s/%s", bucket, key)
  except ClientError as e:
    logger.error("Failed to save analysis to S3: %s", e)
    raise


def handler(event, context):
  """AWS Lambda entry point triggered by an S3 upload event."""
  logger.info("Received event: %s", json.dumps(event))

  # Extract bucket and object key from the event
  try:
    records = event.get("Records", [])
    if not records:
      raise KeyError("No Records in event")

    s3_info = records[0]["s3"]
    bucket = s3_info["bucket"]["name"]
    key = s3_info["object"]["key"]
  except KeyError as e:
    logger.error("Malformed event structure: %s", e)
    raise

  # Generate presigned URL
  presigned_url = generate_presigned_url(bucket, key)

  # Call GPT Action
  # result = call_gpt_action_with_presigned_url(presigned_url, key)
  # analysis_text = result
  # if analysis_text is None:
  #   logger.error("GPT Action response missing 'analysis' field: %s", result)
  #   raise RuntimeError("Missing analysis in GPT Action response")
  #
  # # Build analysis object key
  date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M")
  filename = os.path.basename(key)
  analysis_key = f"analysis/{date_str}-{filename}.md"
  #
  # # Save analysis result to S3
  # save_analysis(bucket, analysis_key, analysis_text)

  return {"statusCode": 200, "body": json.dumps({"analysis_key": analysis_key})}
