"""Minimal OpenRouter client tailored for benchmarking."""

import os
import time
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv


class OpenRouterClient:
    """Lightweight wrapper around the OpenRouter Chat Completions API."""

    load_dotenv()

    def __init__(self, model: str):
        self._model = model
        
        # Check for Swiss AI models (Apertus)
        if model.startswith("swiss-ai/"):
            api_key = os.getenv("SWISSAI_API_KEY")
            if not api_key:
                raise ValueError("SWISSAI_API_KEY not found in environment variables. Please add it to your .env file.")
            self._api_key = api_key
            self._base_url = "https://api.swissai.cscs.ch/v1"
        else:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY not found in environment variables")
            self._api_key = api_key
            self._base_url = "https://openrouter.ai/api/v1"

    @staticmethod
    def _strip_markdown_fences(response: str) -> str:
        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        # Remove common OpenRouter wrappers (e.g., <|begin_of_box|>...<|end_of_box|>)
        wrappers = [
            "<|begin_of_box|>",
            "<|end_of_box|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
        ]
        for token in wrappers:
            cleaned = cleaned.replace(token, "")

        # Drop hidden reasoning blocks such as <think>...</think>
        cleaned = OpenRouterClient._strip_tag_block(cleaned, "<think>", "</think>")

        return cleaned.strip()

    @staticmethod
    def _strip_tag_block(text: str, start_tag: str, end_tag: str) -> str:
        """Remove every occurrence of start_tag ... end_tag (inclusive)."""
        while True:
            start_idx = text.find(start_tag)
            if start_idx == -1:
                break
            end_idx = text.find(end_tag, start_idx + len(start_tag))
            if end_idx == -1:
                break
            text = text[:start_idx] + text[end_idx + len(end_tag):]
        return text

    def chat(self, messages: List[Dict[str, str]], model: Optional[str] = None, retries: int = 5, backoff_factor: float = 2.0) -> str:
        for attempt in range(retries):
            try:
                response = requests.post(
                    f"{self._base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model or self._model,
                        "messages": messages,
                    },
                    timeout=1000,
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                return self._strip_markdown_fences(content)
            except (requests.exceptions.RequestException, requests.exceptions.ChunkedEncodingError) as e:
                if attempt == retries - 1:
                    raise e
                
                wait_time = backoff_factor ** attempt
                print(f"Request failed: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
