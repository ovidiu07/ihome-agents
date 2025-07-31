import boto3
import json
import logging
import openai
import os
import requests
from botocore.exceptions import ClientError
from datetime import datetime, timezone
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from openai import OpenAI

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


def load_system_instructions() -> str:
  try:
    obj = s3_client.get_object(Bucket="devtailor-transactions",
                               Key="gpt/instructions.txt")
    return obj["Body"].read().decode("utf-8")
  except ClientError as e:
    logger.error("Could not fetch instructions from S3: %s", e)
    return "Default fallback instructions here..."


def call_gpt_action_with_presigned_url(file_url: str, filename: str) -> str:
  """Call the OpenAI o3 model with system instructions and JSON content directly."""
  client = OpenAI()
  SYSTEM_INSTRUCTIONS = load_system_instructions()
  print(f"Presigned URL is : {file_url}")
  # Fetch the actual content of the file
  response = requests.get(file_url)
  if response.status_code != 200:
    raise RuntimeError(f"Failed to fetch file content: {response.status_code}")

  json_content = response.text
  messages = [{"role": "system", "content": SYSTEM_INSTRUCTIONS},
    {"role": "user", "content": (f"Here is the JSON file named {filename}:\n\n"
                                 f"{json_content}\n\n"
                                 "Please parse this JSON and produce:\n"
                                 "SECTION 1 — JSON per schema\n"
                                 "SECTION 2 — ~650‑word trading plan\n"
                                 "SECTION 3 — intraday execution bullet plan\n"
                                 "Do not add anything else.")}]

  response = client.chat.completions.create(model="o3", messages=messages)
  logger.info("o3 model responded with finish_reason=%s",
              response.choices[0].finish_reason)
  return response.choices[0].message.content


def call_gpt_action_with_json_content(results: dict, filename: str) -> str:
  """Call the OpenAI o3 model with system instructions and JSON content directly."""
  client = OpenAI()
  SYSTEM_INSTRUCTIONS = load_system_instructions()
  json_content = json.dumps(results, indent=2)
  messages = [{"role": "system", "content": SYSTEM_INSTRUCTIONS},
    {"role": "user", "content": (f"Here is the JSON file named {filename}:\n\n"
                                 f"{json_content}\n\n"
                                 "Please parse this JSON and produce:\n"
                                 "SECTION 1 — JSON per schema\n"
                                 "SECTION 2 — ~650‑word trading plan\n"
                                 "SECTION 3 — intraday execution bullet plan\n"
                                 "Do not add anything else.")}]

  response = client.chat.completions.create(model="o3", messages=messages)
  logger.info("o3 model responded with finish_reason=%s",
              response.choices[0].finish_reason)
  return response.choices[0].message.content


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
  date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
  filename = os.path.basename(key)
  analysis_key = f"analysis/{date_str}-{filename}.md"
  #
  # # Save analysis result to S3
  # save_analysis(bucket, analysis_key, analysis_text)

  return {"statusCode": 200, "body": json.dumps({"analysis_key": analysis_key})}
