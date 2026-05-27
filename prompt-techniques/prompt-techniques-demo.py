import os
import boto3
import certifi
from dotenv import load_dotenv, find_dotenv
from botocore.exceptions import ClientError

_ = load_dotenv(find_dotenv())

client = boto3.client(
    "bedrock-runtime",
    region_name=os.getenv("AWS_REGION", "us-west-1"),
    verify=certifi.where()
)

# model_id = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
model_id = "us.anthropic.claude-sonnet-4-6"

import importlib.util

## Change this to the desired prompt module name (without .py extension)
prompt_module = "expanding2"  # e.g., "expanding2" or "chatbots1"
spec = importlib.util.spec_from_file_location(
    prompt_module, os.path.join("prompts", f"{prompt_module}.py")
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
prompt = mod.prompt

conversation = [
    {
        "role": "user",
        "content": [{"text": prompt}],
    }
]

# --- Example 1: Direct (non-streaming) response ---
print("=== Direct Response ===")
try:
    response = client.converse(
        modelId=model_id,
        messages=conversation,
        inferenceConfig={"maxTokens": 512, "temperature": 0.0},
    )
    # print(f"Raw Response: {response}")
    response_text = response["output"]["message"]["content"][0]["text"]
    print(response_text)
except (ClientError, Exception) as e:
    print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
    exit(1)

## --- Example 2: Streaming response ---
# print("\n=== Streaming Response ===")
# try:
#     response = client.converse_stream(
#         modelId=model_id,
#         messages=conversation,
#         inferenceConfig={"maxTokens": 512, "temperature": 0.5},
#     )
#     for event in response["stream"]:
#         if "contentBlockDelta" in event:
#             delta = event["contentBlockDelta"]["delta"]
#             if "text" in delta:
#                 print(delta["text"], end="", flush=True)
#     print()  # newline after stream ends
# except (ClientError, Exception) as e:
#     print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
#     exit(1)
