from dotenv import load_dotenv
from functools import lru_cache
import logging
import os
import requests

load_dotenv('../.env')

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

"""
Anthropic API documentation:
https://docs.anthropic.com/claude/docs
"""

# Fallback models if API listing fails
ANTHROPIC_FALLBACK_MODELS = [
    "claude-3-5-sonnet-20240620",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    "claude-2.1",
    "claude-2.0",
    "claude-instant-1.2",
]


def get_anthmodlst() -> list[str]:
    """
    Gets list of Anthropic models from the API.
    Returns list of model IDs.
    """
    ANTHROPIC_API_BASE = "https://api.anthropic.com/v1"
    
    if not ANTHROPIC_API_KEY:
        return ["ANTHROPIC_API_KEY not set"]
    
    headers = {
        "Authorization": f"Bearer {ANTHROPIC_API_KEY}",
        "Anthropic-Version": "2023-06-01",
        "Content-Type": "application/json",
    }
    
    try:
        r = requests.get(
            f"{ANTHROPIC_API_BASE}/models",
            headers=headers,
            timeout=(5, 10),
        )
        r.raise_for_status()
        data = r.json()
        results = [m["id"] for m in data.get("data", [])]
        if not results:
            results = ["(No models returned)"]
    except Exception as e:
        results = [f"Failed to list Anthropic models: {e}"]
        logging.warning(f"Failed to list Anthropic models: {e}", exc_info=True)
    
    return results


def get_anthmodcostlst() -> list[tuple[str, float, float]]:
    """
    Gets list of Anthropic models along with per-token costs.
    Returns list of tuples: (model_id, input_cost_per_million, output_cost_per_million)
    """
    ANTHROPIC_API_BASE = "https://api.anthropic.com/v1"
    results = []
    
    if not ANTHROPIC_API_KEY:
        return [("ANTHROPIC_API_KEY not set", 0.0, 0.0)]
    
    headers = {
        "Authorization": f"Bearer {ANTHROPIC_API_KEY}",
        "Anthropic-Version": "2023-06-01",
        "Content-Type": "application/json",
    }
    
    try:
        r = requests.get(
            f"{ANTHROPIC_API_BASE}/models",
            headers=headers,
            timeout=(5, 10),
        )
        r.raise_for_status()
        data = r.json()
        
        # Anthropic's API doesn't return pricing in the models endpoint
        # We'll use known pricing for common models
        pricing_map = {
            "claude-3-5-sonnet-20240620": (3.00, 15.00),    # $3 input, $15 output per million tokens
            "claude-3-opus-20240229": (15.00, 75.00),      # $15 input, $75 output per million tokens  
            "claude-3-sonnet-20240229": (3.00, 15.00),    # $3 input, $15 output per million tokens
            "claude-3-haiku-20240307": (0.25, 1.25),       # $0.25 input, $1.25 output per million tokens
            "claude-2.1": (8.00, 24.00),                  # $8 input, $24 output per million tokens
            "claude-2.0": (8.00, 24.00),                  # $8 input, $24 output per million tokens
            "claude-instant-1.2": (0.80, 2.40),           # $0.80 input, $2.40 output per million tokens
        }
        
        for m in data.get("data", []):
            model_id = m.get("id", "(unknown)")
            input_cost, output_cost = pricing_map.get(model_id, (0.0, 0.0))
            results.append((model_id, input_cost, output_cost))
        
        if not results:
            results = [("(No models returned)", 0.0, 0.0)]
            
    except Exception as e:
        results = [(f"Failed to list Anthropic models: {e}", 0.0, 0.0)]
        logging.warning(f"Failed to list Anthropic models: {e}", exc_info=True)
    
    return results


@lru_cache(maxsize=1)
def get_anthropic_models() -> list[str]:
    """
    Cached list of available Anthropic models.
    """
    models = get_anthmodlst()
    
    # If we get a single error message, use fallback list
    if len(models) == 1 and not models[0].startswith("claude"):
        logging.warning(f"get_anthropic_models: {models[0]}")
        logging.warning("get_anthropic_models: using hardcoded fallback list")
        return sorted(ANTHROPIC_FALLBACK_MODELS)
    
    # Filter to only Claude models
    return sorted(
        m for m in models
        if "claude" in m.lower()
    )