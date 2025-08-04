from __future__ import annotations
import boto3
import json, re, logging, time
import logging
import openai
import os
import requests
# Email sending function
import smtplib
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


def load_system_instructions(is_general_analysis: bool = True) -> str:
  """Load system instructions for the GPT call from S3."""
  try:
    key = (
      "gpt/instructions.txt" if is_general_analysis else "gpt/intraday_instructions.txt")
    obj = s3_client.get_object(Bucket="devtailor-transactions", Key=key)
    return obj["Body"].read().decode("utf-8")
  except ClientError as e:
    logger.error("Could not fetch instructions from S3: %s", e)
    return "Default fallback instructions here..."

# ---------------------------------------------------------------------------
# Simple markdown-based validator ― tighten as needed
# ---------------------------------------------------------------------------
_MD_SCHEME = {
  "SECTION 1 —": "step-by-step trading plan",
  "SECTION 2 —": "intraday execution plan",
}

_PAT_5MIN_TABLE = re.compile(
    r"#### 5[- ]Minute Forecast[^\n]*\n\n\| UTC[^\n]+\|", re.I
)

def _validates(markdown: str) -> bool:
  """Check if all headings are present and the 5-min table exists."""
  for header, trailer in _MD_SCHEME.items():
    if header not in markdown or trailer not in markdown:
      logger.debug("Missing header: %s%s", header, trailer)
      return False
  if not _PAT_5MIN_TABLE.search(markdown):
    logger.debug("Missing 5-minute forecast table")
    return False
  return True

# ---------------------------------------------------------------------------
# Core wrapper
# ---------------------------------------------------------------------------
def call_gpt_action_with_json_content(
    results: Dict,
    filename: str,
    is_general_analysis: bool = True,
    previous_report: str | None = None,
    max_retries: int = 1,
) -> str:
  """
  Call OpenAI with system + user messages and JSON payload.
  Enforce deterministic sampling for intraday (gpt-4o-mini) and
  schema-validate the response.
  """
  system_prompt = load_system_instructions(is_general_analysis)
  json_content = json.dumps(results, indent=2)

  messages: List[Dict[str, str]] = [
    {"role": "system", "content": system_prompt}
  ]

  # Add previous report context for intraday updates
  if not is_general_analysis and previous_report:
    messages.append({
      "role": "user",
      "content": (
        "Here is the previous general analysis report:\n\n"
        f"{previous_report}\n\n"
        "Update this report as instructed using the intraday data."
      )
    })

  # Always attach the current JSON snapshot
  messages.append({
    "role": "user",
    "content": (
      f"Here is the JSON file named {filename}:\n\n"
      f"{json_content}\n\n"
      "Please parse this JSON and produce sections as specified. "
      "Do not add anything else."
    )
  })

  # ---------------------------------------------------------------------
  # Model-specific parameters
  # ---------------------------------------------------------------------
  if is_general_analysis:
    model_name = "o3"
    kwargs = {}                 # server defaults (temp≈1)
  else:
    model_name = "gpt-4o-mini"
    kwargs = {
      "temperature": 0,
      "top_p": 0,
      "seed": 42           # supported by v2 chat API
    }

  # ---------------------------------------------------------------------
  # Straightforward retry loop with schema validation
  # ---------------------------------------------------------------------
  for attempt in range(max_retries + 1):
    try:
      resp = client.chat.completions.create(
          model=model_name,
          messages=messages,
          **kwargs
      )
      content = resp.choices[0].message.content
      if _validates(content):
        send_email_with_analysis(content, filename)
        logger.info(
            "Model responded ok (finish_reason=%s)",
            resp.choices[0].finish_reason,
        )
        return content

      logger.warning("Schema validation failed on attempt %d", attempt + 1)
    except OpenAIError as e:
      logger.error("OpenAI API error: %s", e)
    # brief back-off before retry
    if attempt < max_retries:
      time.sleep(1.5)

  # If we reach here: unrecoverable failure
  raise RuntimeError("Assistant failed to return a valid report after retries.")


def send_email_with_analysis(content: str, subject_filename: str):
  sender_email = os.getenv("SENDER_EMAIL") or "contact@ihomeprosolutions.ro"
  receiver_raw = os.getenv("RECEIVER_EMAIL") or "moldovan.ovidiuv@gmail.com, moldovan.iuliae@gmail.com"
  smtp_server = os.getenv("SMTP_SERVER") or "smtppro.zoho.eu"
  smtp_port = int(os.getenv("SMTP_PORT", 587))
  smtp_username = os.getenv("SMTP_USERNAME") or "contact@ihomeprosolutions.ro"
  smtp_password = os.getenv("SMTP_PASSWORD") or "Marley01042022$"

  receiver_list = [email.strip() for email in receiver_raw.split(",") if email.strip()]

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
