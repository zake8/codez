#!/usr/bin/env python3
"""
Anthropic Models Utility
========================

A utility for retrieving Anthropic model information and pricing using the official
Anthropic Python SDK. Provides functions to list available models,
retrieve (stale) pricing information, and get detailed model capabilities.

Usage:
>>> from anthropic_stuff import get_anthropic_models, get_anthmodcostlst
>>> 
>>> # Get list of model IDs
>>> models = get_anthropic_models()
>>> print(models[:3])  # Show first 3 models
['claude-3-5-sonnet-20240620', 'claude-3-opus-20240229', ...]
>>> 
>>> # Get models with pricing
>>> models_with_costs = get_anthmodcostlst()
>>> for model_id, input_cost, output_cost in models_with_costs:
...     print(f"{model_id}: Input ${input_cost}/M, Output ${output_cost}/M")

Pricing is hardcoded based on Anthropic's published rates (https://www.anthropic.com/pricing).

Handles missing API keys by returning error messages and falling back
to a hardcoded list of common models.
"""

from dotenv import load_dotenv
from functools import lru_cache
import logging
import os
import anthropic

load_dotenv('../.env')

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# Known pricing for Anthropic models (per million tokens)
# Source: https://www.anthropic.com/pricing
ANTHROPIC_PRICING_MAP = {
    "claude-3-5-sonnet-20240620": (3.00, 15.00),    # $3 input, $15 output
    "claude-3-opus-20240229": (15.00, 75.00),      # $15 input, $75 output
    "claude-3-sonnet-20240229": (3.00, 15.00),    # $3 input, $15 output
    "claude-3-haiku-20240307": (0.25, 1.25),       # $0.25 input, $1.25 output
    "claude-2.1": (8.00, 24.00),                  # $8 input, $24 output
    "claude-2.0": (8.00, 24.00),                  # $8 input, $24 output
    "claude-instant-1.2": (0.80, 2.40),           # $0.80 input, $2.40 output
    # Newer models from the sample
    "claude-opus-5": (15.00, 75.00),              # Opus 5 pricing (estimated)
    "claude-sonnet-5": (3.00, 15.00),             # Sonnet 5 pricing (estimated)
    "claude-fable-5": (1.00, 5.00),               # Fable 5 pricing (estimated)
    "claude-opus-4-8": (15.00, 75.00),            # Opus 4.8 pricing
    "claude-opus-4-7": (15.00, 75.00),            # Opus 4.7 pricing
    "claude-sonnet-4-6": (3.00, 15.00),           # Sonnet 4.6 pricing
    "claude-opus-4-6": (15.00, 75.00),            # Opus 4.6 pricing
    "claude-opus-4-5-20251101": (15.00, 75.00),  # Opus 4.5 pricing
    "claude-haiku-4-5-20251001": (0.25, 1.25),   # Haiku 4.5 pricing
    "claude-sonnet-4-5-20250929": (3.00, 15.00),  # Sonnet 4.5 pricing
}

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
    Get list of Anthropic model IDs from the API.
    
    Uses the official Anthropic Python SDK to fetch available models.
    Returns a list of model IDs (strings).
    
    Returns:
        list[str]: List of model IDs
        
    Example:
        >>> models = get_anthmodlst()
        >>> print(models[:3])
        ['claude-3-5-sonnet-20240620', 'claude-3-opus-20240229', ...]
        
    Note:
        If API key is missing or API call fails, returns error message in list.
        Falls back to hardcoded model list when single error is returned.
    """
    if not ANTHROPIC_API_KEY:
        return ["ANTHROPIC_API_KEY not set"]
    
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        page = client.models.list()
        
        results = []
        for model in page.data:
            if model.type == 'model':  # Only include actual models, not other types
                results.append(model.id)
        
        if not results:
            results = ["(No models returned)"]
            
        return results
        
    except Exception as e:
        results = [f"Failed to list Anthropic models: {e}"]
        logging.warning(f"Failed to list Anthropic models: {e}", exc_info=True)
        return results


def get_anthmodcostlst() -> list[tuple[str, float, float]]:
    """
    Get list of Anthropic models with pricing information.
    
    Returns a list of tuples containing (model_id, input_cost_per_million, 
    output_cost_per_million). Pricing is based on Anthropic's published rates.
    
    Returns:
        list[tuple[str, float, float]]: List of (model_id, input_cost, output_cost) tuples
        
    Example:
        >>> models_with_costs = get_anthmodcostlst()
        >>> for model_id, input_cost, output_cost in models_with_costs:
        ...     print(f"{model_id}: Input ${input_cost}/M, Output ${output_cost}/M")
    """
    if not ANTHROPIC_API_KEY:
        return [("ANTHROPIC_API_KEY not set", 0.0, 0.0)]
    
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        page = client.models.list()
        
        results = []
        for model in page.data:
            if model.type == 'model':
                model_id = model.id
                input_cost, output_cost = ANTHROPIC_PRICING_MAP.get(model_id, (0.0, 0.0))
                results.append((model_id, input_cost, output_cost))
        
        if not results:
            results = [("(No models returned)", 0.0, 0.0)]
            
        return results
        
    except Exception as e:
        results = [(f"Failed to list Anthropic models: {e}", 0.0, 0.0)]
        logging.warning(f"Failed to list Anthropic models: {e}", exc_info=True)
        return results


@lru_cache(maxsize=1)
def get_anthropic_models() -> list[str]:
    """
    Get cached list of available Claude models (filtered and sorted).
    
    Returns a cached, filtered list of Claude models. Uses LRU cache for performance.
    Filters to only include models with 'claude' in the name.
    
    Returns:
        list[str]: Sorted list of Claude model IDs
        
    Example:
        >>> models = get_anthropic_models()
        >>> print(len(models))
        15
        
    Note:
        This is the recommended function for most use cases as it provides
        a clean, filtered list of Claude models with caching for performance.
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


def get_model_details() -> list[dict]:
    """
    Get detailed information about Anthropic models.
    
    Returns comprehensive model information including capabilities, token limits,
    creation dates, and pricing. This is the most detailed function providing
    everything needed for model selection interfaces.
    
    Returns:
        list[dict]: List of model dictionaries with keys:
            - id: Model ID string
            - display_name: Human-readable model name
            - max_input_tokens: Maximum input tokens
            - max_output_tokens: Maximum output tokens
            - created_at: Creation date (ISO format string)
            - input_cost_per_million: Input cost per million tokens
            - output_cost_per_million: Output cost per million tokens
            - capabilities: Dictionary of boolean capabilities
        
    Example:
        >>> details = get_model_details()
        >>> for model in details:
        ...     print(f"{model['display_name']}: {model['max_input_tokens']} tokens")
        
    Note:
        Capabilities include: batch, citations, code_execution, image_input,
        pdf_input, structured_outputs, thinking.
    """
    if not ANTHROPIC_API_KEY:
        return [{"error": "ANTHROPIC_API_KEY not set"}]
    
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        page = client.models.list()
        
        results = []
        for model in page.data:
            if model.type == 'model':
                model_info = {
                    'id': model.id,
                    'display_name': model.display_name,
                    'max_input_tokens': model.max_input_tokens,
                    'max_output_tokens': model.max_tokens,
                    'created_at': model.created_at.isoformat() if hasattr(model.created_at, 'isoformat') else str(model.created_at),
                    'input_cost_per_million': ANTHROPIC_PRICING_MAP.get(model.id, (0.0, 0.0))[0],
                    'output_cost_per_million': ANTHROPIC_PRICING_MAP.get(model.id, (0.0, 0.0))[1],
                    'capabilities': {
                        'batch': model.capabilities.batch.supported if hasattr(model.capabilities, 'batch') else False,
                        'citations': model.capabilities.citations.supported if hasattr(model.capabilities, 'citations') else False,
                        'code_execution': model.capabilities.code_execution.supported if hasattr(model.capabilities, 'code_execution') else False,
                        'image_input': model.capabilities.image_input.supported if hasattr(model.capabilities, 'image_input') else False,
                        'pdf_input': model.capabilities.pdf_input.supported if hasattr(model.capabilities, 'pdf_input') else False,
                        'structured_outputs': model.capabilities.structured_outputs.supported if hasattr(model.capabilities, 'structured_outputs') else False,
                        'thinking': model.capabilities.thinking.supported if hasattr(model.capabilities, 'thinking') else False,
                    }
                }
                results.append(model_info)
        
        return results
        
    except Exception as e:
        return [{"error": f"Failed to get model details: {e}"}]


if __name__ == "__main__":
    # Example usage
    print("=== Available Models ===")
    models = get_anthropic_models()
    for model in models:
        print(f"- {model}")
    
    print("\n=== Models with Costs ===")
    models_with_costs = get_anthmodcostlst()
    for model_id, input_cost, output_cost in models_with_costs:
        print(f"{model_id}: Input ${input_cost}/M, Output ${output_cost}/M")
    
    print("\n=== Detailed Model Information ===")
    details = get_model_details()
    for detail in details:
        if 'error' in detail:
            print(f"Error: {detail['error']}")
        else:
            print(f"\nModel: {detail['display_name']} ({detail['id']})")
            print(f"  Input Tokens: {detail['max_input_tokens']:,}")
            print(f"  Output Tokens: {detail['max_output_tokens']:,}")
            print(f"  Cost: Input ${detail['input_cost_per_million']}/M, Output ${detail['output_cost_per_million']}/M")
            print(f"  Created: {detail['created_at']}")
            print("  Capabilities:")
            for cap, value in detail['capabilities'].items():
                print(f"    {cap}: {'✓' if value else '✗'}")