from dotenv import load_dotenv
from functools import lru_cache
import logging
import os
from gradient import Gradient

load_dotenv('../.env')
##### GRADIENT_MODEL_ACCESS_KEY = os.environ.get("GRADIENT_MODEL_ACCESS_KEY")
DIGITALOCEAN_API_TOKEN = os.environ.get("DIGITALOCEAN_API_TOKEN")


DIGITALOCEAN_FALLBACK_MODELS = [
    "anthropic-claude-3.5-haiku",
    "anthropic-claude-4.5-haiku",
    "anthropic-claude-haiku-4.5",
    "anthropic-claude-3.5-sonnet",
    "anthropic-claude-3.7-sonnet",
    "anthropic-claude-sonnet-4",
    "anthropic-claude-4.5-sonnet",
    "anthropic-claude-4.6-sonnet",
    "anthropic-claude-3-opus",
    "anthropic-claude-opus-4",
    "anthropic-claude-4.1-opus",
    "anthropic-claude-opus-4.5",
]


anthropic_dog_info = """
## Anthropic’s model lineup:
- Haiku:  Fast, lightweight, short answers - High-volume app
- Sonnet: Balanced reasoning + cost - Coding assistant / balanced app
- Opus:   Deep reasoning, complex tasks - Complex reasoning / architecture planning / multi-step
- v4.6:   iteration improvement; Usually better reasoning, fewer hallucinations; Slightly improved tool handling

## typical Anthropic specifications:
- Claude 3 Opus / Claude 3.5 Sonnet / Claude 3.7 Sonnet: 200,000 tokens
- Claude 3.5 Haiku: 200,000 tokens
- Claude 4 / 4.1 / 4.5 / 4.6 variants (Opus, Sonnet, Haiku): Likely 200,000 tokens (Anthropic has standardized on 200k for recent models)

## costs:
- Haiku 3.5:      input_rate =  0.80; output_rate =  4.00
- Haiku 4.5:      input_rate =  1.00; output_rate =  5.00
- Sonnet:         input_rate =  3.00; output_rate = 15.00
- Opus 3, 4, 4.1: input_rate = 15.00; output_rate = 75.00
- Opus 4.5, 4.6:  input_rate =  5.00; output_rate = 25.00
"""


def get_domodlst() -> list[str]:
    """Gets list of DigitalOcean Gradient models"""
    if not DIGITALOCEAN_API_TOKEN:  ### should do this check elsewhere
        return ["DIGITALOCEAN_API_TOKEN not set"]
    try:
        client = Gradient(access_token=DIGITALOCEAN_API_TOKEN)
        models_resp = client.models.list()
        return [m.id for m in models_resp.models]
    except Exception as e:
        mess = f"Failed to list DigitalOcean Gradient models: {e}"
        logging.warning(mess, exc_info=True)
        return [mess]


@lru_cache(maxsize=1)
def get_digitalocean_models() -> list[str]:
    """
    Cached list of available DigitalOcean Gradient models.
    """
    models = get_domodlst()
    if len(models) == 1 and not models[0].startswith("anthropic"):
        logging.warning(f"get_digitalocean_models: {models[0]}")
        logging.warning("get_digitalocean_models: using hardcoded fallback list")
        return sorted(DIGITALOCEAN_FALLBACK_MODELS)
    return sorted(
        m for m in models
        if "claude" in m.lower()
    )
