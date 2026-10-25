"""Temporary Qwen Model Studio client for image descriptions."""
import base64
import mimetypes
import os
from pathlib import Path
from typing import Optional

from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_URL = "https://token-plan.maas.qwencloudapi.com/compatible-mode/v1"
MODEL = "qwen3.8-max"
_KEY = os.environ.get("DASHSCOPE_API_KEY", "").strip()
_client_instance: Optional[OpenAI] = None


def _client() -> OpenAI:
    global _client_instance
    if _client_instance is None:
        if not _KEY:
            raise RuntimeError(
                "DASHSCOPE_API_KEY is not set. Add your Model Studio API key "
                "to the environment or submission_repo/.env."
            )
        _client_instance = OpenAI(api_key=_KEY, base_url=BASE_URL)
    return _client_instance


def describe_image(image_path: str, prompt: str) -> str:
    """Send an image file and prompt to Qwen and return the response."""
    path = Path(image_path)
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return describe_image_bytes(path.read_bytes(), mime, prompt)


def describe_image_bytes(image_bytes: bytes, mime: str, prompt: str) -> str:
    """Send raw image bytes and prompt to Qwen; thread-safe, no temp file."""
    image_data = base64.b64encode(image_bytes).decode("ascii")

    stream = _client().chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{image_data}"},
                },
            ],
        }],
        extra_body={"enable_thinking": True},
        stream=True,
    )

    response_parts = []
    for chunk in stream:
        if not chunk.choices:
            continue
        content = chunk.choices[0].delta.content
        if content:
            response_parts.append(content)

    description = "".join(response_parts).strip()
    if not description:
        raise RuntimeError("Qwen image API returned no description.")
    return description
