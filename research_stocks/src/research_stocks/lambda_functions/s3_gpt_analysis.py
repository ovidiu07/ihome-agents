import json
import logging
import os
from datetime import datetime, timezone
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import boto3
import requests
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)
app = FastAPI()
s3_client = boto3.client("s3")

def generate_presigned_url(bucket: str, key: str, expiration: int = 300) -> str:
    """Generate a presigned URL to access the uploaded S3 object."""
    try:
        url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=expiration,
        )
        logger.info("Generated presigned URL for %s/%s", bucket, key)
        return url
    except ClientError as e:
        logger.error("Failed to generate presigned URL: %s", e)
        raise

def call_gpt_action(file_url: str, filename: str) -> dict:
    """Call the GPT Action endpoint with the provided file URL and filename."""
    BASE_URL = "https://go-sweet.ro"
    endpoint = "/agent-analysis"
    if not endpoint:
        raise RuntimeError("GPT_ACTION_URL environment variable not set")

    payload = {"file_url": file_url, "filename": filename}
    try:
        response = requests.post(BASE_URL+endpoint, json=payload, timeout=30)
        response.raise_for_status()
        logger.info("GPT Action responded with status %s", response.status_code)
        return response.json()
    except requests.RequestException as e:
        logger.error("Request to GPT Action failed: %s", e)
        raise

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
    result = call_gpt_action(presigned_url, key)
    analysis_text = result.get("analysis")
    if analysis_text is None:
        logger.error("GPT Action response missing 'analysis' field: %s", result)
        raise RuntimeError("Missing analysis in GPT Action response")

    # Build analysis object key
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    filename = os.path.basename(key)
    analysis_key = f"analysis/{date_str}-{filename}.md"

    # Save analysis result to S3
    save_analysis(bucket, analysis_key, analysis_text)

    return {
        "statusCode": 200,
        "body": json.dumps({"analysis_key": analysis_key})
    }
