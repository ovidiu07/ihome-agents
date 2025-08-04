import boto3
import json
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


def call_gpt_action_with_json_content(results: dict, filename: str,
    is_general_analysis: bool = True,
    previous_report: str | None = None, ) -> str:
  """Call the OpenAI model with system instructions and JSON content."""
  client = OpenAI()
  SYSTEM_INSTRUCTIONS = load_system_instructions(is_general_analysis)
  json_content = json.dumps(results, indent=2)
  messages = [{"role": "system", "content": SYSTEM_INSTRUCTIONS}]
  # Optional: if intraday and previous general analysis exists
  if not is_general_analysis and previous_report:
    messages.append({"role": "user", "content": (
      "Here is the previous general analysis report:\n\n"
      f"{previous_report}\n\n"
      "Update this report as mentioned in instructions according to intradaily data.")})

  # Always send the JSON content
  messages.append({"role": "user",
                   "content": (f"Here is the JSON file named {filename}:\n\n"
                               f"{json_content}\n\n"
                               "Please parse this JSON and produce sections as mentioned in instructions:\n"
                               "Do not add anything else.")})
  model_name = "o3" if is_general_analysis else "gpt-4o-mini"
  response = client.chat.completions.create(model=model_name, messages=messages)
  send_email_with_analysis(response.choices[0].message.content, filename)
  logger.info("Model responded with finish_reason=%s",
              response.choices[0].finish_reason)
  return response.choices[0].message.content


def send_email_with_analysis(content: str, subject_filename: str):
  sender_email = os.getenv("SENDER_EMAIL")
  receiver_raw = os.getenv("RECEIVER_EMAIL")
  smtp_server = os.getenv("SMTP_SERVER")
  smtp_port = int(os.getenv("SMTP_PORT", 587))
  smtp_username = os.getenv("SMTP_USERNAME")
  smtp_password = os.getenv("SMTP_PASSWORD")
  receiver_list = [email.strip() for email in receiver_raw.split(",") if email.strip()]
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

      logger.info("Sent analysis email to %s", receiver_list)
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
