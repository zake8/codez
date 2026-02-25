from dotenv import load_dotenv
from flask import render_template
from functools import lru_cache
import logging
import os
import requests

load_dotenv('../.env')

DEVSTRAL_API_KEY = os.environ.get("DEVSTRAL_API_KEY")

"""
| Temperature   | Behavior                         |
| ------------- | -------------------------------- |
| **0.0 – 0.3** | Highly deterministic, accurate   |
| **0.3 – 0.7** | Balanced creativity + coherence  |
| **0.7 – 1.0** | Highly creative, but less stable |
0.2 will make the model more focused and consistent.
https://platform-docs-public.pages.dev/api/?utm_source=chatgpt.com
https://docs.mistral.ai/getting-started/glossary#temperature
"""

DEVSTRAL_FALLBACK_MODELS = [
    "devstral-small-2507",
    "devstral-medium-2507",
    "devstral-2512",
    "devstral-medium-latest",
    "devstral-latest",
    "labs-devstral-small-2512",
    "devstral-small-latest",
]


def get_mismodlst() -> list[str]:
    """ Gets list of Mistral models """
    MISTRAL_API_BASE = "https://api.mistral.ai/v1"
    if not DEVSTRAL_API_KEY:
        results = ["DEVSTRAL_API_KEY not set"]
    else:
        headers = {
            "Authorization": f"Bearer {DEVSTRAL_API_KEY}",
        }
        try:
            r = requests.get(
                f"{MISTRAL_API_BASE}/models",
                headers=headers,
                timeout=(5, 10),
            )
            r.raise_for_status()
            data = r.json()
            results = [m["id"] for m in data.get("data", [])]
            if not results:
                results = ["(No models returned)"]
        except Exception as e:
            results = [f"Failed to list Mistral models: {e}"]
    return results


def get_mismodcostlst() -> list[tuple[str, float, float]]:
    """Gets list of Mistral models along with per-token costs."""
    MISTRAL_API_BASE = "https://api.mistral.ai/v1"
    results = []
    if not DEVSTRAL_API_KEY:
        return [("DEVSTRAL_API_KEY not set", 0.0, 0.0)]
    headers = {
        "Authorization": f"Bearer {DEVSTRAL_API_KEY}",
    }
    try:
        r = requests.get(
            f"{MISTRAL_API_BASE}/models",
            headers=headers,
            timeout=(5, 10),
        )
        r.raise_for_status()
        data = r.json()
        for m in data.get("data", []):
            model_id = m.get("id", "(unknown)")
            pricing = m.get("pricing", {})
            prompt_cost = pricing.get("prompt", 0.0)
            completion_cost = pricing.get("completion", 0.0)
            results.append((model_id, prompt_cost, completion_cost))
        if not results:
            results = [("(No models returned)", 0.0, 0.0)]
    except Exception as e:
        results = [(f"Failed to list Mistral models: {e}", 0.0, 0.0)]
    return results


@lru_cache(maxsize=1)
def get_devstral_models() -> list[str]:
    """
    Cached list of available Devstral models.
    """
    models = get_mismodlst()
    if len(models) == 1 and not models[0].startswith("devstral"):
        logging.warning(f"get_devstral_models: {models[0]}")
        logging.warning("get_devstral_models: using hardcoded fallback list")
        return sorted(DEVSTRAL_FALLBACK_MODELS)
    return sorted(
        m for m in models
        if "devstral" in m.lower()
    )
