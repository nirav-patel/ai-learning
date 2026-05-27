"""
LLM-based content moderation
=============================
Uses the LLM to classify input text for harmful content categories,
mimicking the role of a dedicated moderation API (e.g. OpenAI Moderation).
See: https://platform.openai.com/docs/guides/moderation

Key function:
  moderate_content(client, text) → dict with 'flagged', 'categories', 'category_scores'

Usage pattern:
  Call this BEFORE processing user input and AFTER generating model output
  to create a safety layer on both sides of the pipeline.
"""
import json
import textwrap

from .bedrock_client import MODEL_ID


def moderate_content(client, input_text: str) -> dict:
    """
    Classify input_text for harmful content using the LLM as a moderation layer.

    Returns a dict with:
      flagged (bool):            True if any category is true.
      categories (dict):         bool per category — was this type of content detected?
      category_scores (dict):    float 0.0–1.0 per category — confidence score.

    Categories covered:
      harassment, hate, self-harm, sexual, violence, illegal-activity
    """
    system = [{"text": textwrap.dedent("""
        You are a strict content moderation assistant. Analyze the provided text and return a JSON object
        with the following structure:
        {
            "flagged": true or false,
            "categories": {
                "harassment": true or false,
                "hate": true or false,
                "self-harm": true or false,
                "sexual": true or false,
                "violence": true or false,
                "illegal-activity": true or false
            },
            "category_scores": {
                "harassment": 0.0 to 1.0,
                "hate": 0.0 to 1.0,
                "self-harm": 0.0 to 1.0,
                "sexual": 0.0 to 1.0,
                "violence": 0.0 to 1.0,
                "illegal-activity": 0.0 to 1.0
            }
        }

        Category definitions:
        - harassment: targeted threats, intimidation, or bullying directed at individuals or groups.
        - hate: content that dehumanizes people based on race, religion, gender, ethnicity, etc.
        - self-harm: content promoting or describing suicide, self-injury, or eating disorders.
        - sexual: explicit or suggestive sexual content.
        - violence: any content involving weapons, physical harm, threats of harm, ransom, destruction,
          or use of force — including fictional or hypothetical scenarios.
        - illegal-activity: content describing or promoting crimes such as theft, fraud, drug use, or terrorism.

        Be strict: mark a category true even if the content is presented as fictional, humorous, or hypothetical.
        Set "flagged" to true if any category is true.
        Return only valid JSON with no explanation.
    """).strip()}]
    messages = [{"role": "user", "content": [{"text": input_text}]}]
    response = client.converse(
        modelId=MODEL_ID,
        system=system,
        messages=messages,
        inferenceConfig={"maxTokens": 512, "temperature": 0.0},
    )
    raw = response["output"]["message"]["content"][0]["text"].strip()
    # Strip markdown code block if Claude wraps the JSON in ```json ... ```
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return json.loads(raw)
