if poss, plz perform Slice 3.2 and provide diffs as needed

==================

# obj: implement enhancement to "codez" (this repo) to use Anthropic direct as third option for "platform"; don't break anything

## Implementation Notes:

- app/routes.py - Main file that needs changes:
    Add Anthropic API key constant / load ANTHROPIC_API_KEY 
    Add new route handler for Anthropic model list
    Add call_anthropic() function similar to call_dog() and call_devstral()
    Update ui() function to handle Anthropic platform choice
      Add Anthropic models to template rendering
      Update cost calculation in get_rates() for Anthropic models
    Update template rendering to include Anthropic options
    New Functions in app/routes.py
       get_anthropic_models() - Cached function to get available models (similar to get_digitalocean_models())
       call_anthropic() - API calling function (similar to call_dog() and call_devstral())
       New route /anthropic_modellst - Model listing endpoint

- app/templates/ui.html - Need to add:
    Add third radio button for Anthropic platform
    Add model dropdown for Anthropic (populated from new endpoint)
    Update conditional rendering for platform-specific options

- app/init.py - May need to add Anthropic-related imports (but likely not)

- clear patterns for -- follow the same pattern for Anthropic:
    DigitalOcean Gradient (call_dog())
    Mistral/Devstral (call_devstral())
- Key functions to add/modify:
    call_anthropic() - Similar to existing API call functions
    New route for /anthropic_modellst to get available models
    Updates to ui() to handle the new platform
- Anthropic API key is available in .env as ANTHROPIC_API_KEY -- 3.1
- Want to use Anthropic's Messages API (likely for single-shot coding w/ Sonnet)
- Use flash messages for user-facing errors
- Graceful fallback if API key missing
- pipenv install anthropic -- 3.1

## slice 1 -- pass to see which files to view
- ui.html 
- __init__.py 
- routes.py

## slice 2 -- create mini design document
- morphed into "Implementation Notes"

## slice 2.5 -- backup critical files!
- ui.html.bak-2026-07-21
- routes.py.bak-2026-07-21

## slice 2.75 -- write out function refactors
- like three or four specific ones
- these will be slice 3.1, 3.2, etc. each 
- can be cross repo code changes; each not too big or complex to overwhelm context

## slice 3 -- Provide the specific code changes for each piece in design one by one, iterating over small refactors


## Slice 3.1: Add Anthropic API Configuration and Constants

File: app/routes.py

# Add to imports section
import anthropic  # Add this import

# Add after existing API key loads
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")


## Slice 3.2: Add get_anthropic_models() Function

File: app/routes.py

def get_anthropic_models() -> list[str]:
    """
    Get available Anthropic models.
    Returns cached list of model IDs.
    """
    return [
    ]

Anthropic provides a Models API for discovering which models are available to your API key.

The endpoint is:

GET https://api.anthropic.com/v1/models

Example:

curl https://api.anthropic.com/v1/models \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01"

It returns a paginated list of models, for example:

{
  "data": [
    {
      "id": "claude-sonnet-4-20250514",
      "display_name": "Claude Sonnet 4",
      "created_at": "2025-05-14T00:00:00Z",
      "max_input_tokens": 200000,
      "max_tokens": 64000,
      "capabilities": {
        ...
      },
      "type": "model"
    }
  ],
  "first_id": "...",
  "last_id": "...",
  "has_more": false
}

The API supports:

limit — number of models to return
after_id / before_id — pagination
optional beta headers for beta capabilities

Anthropic also provides:

GET /v1/models/{model_id} — retrieve details about a specific model or resolve an alias to its canonical model ID.

If you're writing software that targets multiple providers, the Anthropic Models API is conceptually similar to OpenAI's /v1/models, making it straightforward to discover supported models at runtime rather than hard-coding model names.


## Slice 3.3: Add call_anthropic() Function

File: app/routes.py

def call_anthropic(
    prompt_blob: str,
    custom_system_prompt: str,
    model: str = "claude-3-sonnet-20240229",
    temperature: float = 0.2,
    timeout: int = 45,
) -> tuple[str, float, str]:
    """
    Call Anthropic API with given prompt, model, and temperature.
    Returns a tuple:
      - generated_text (Markdown-ready str)
      - cost (float)
      - finish_reason (str)
    """
    if not ANTHROPIC_API_KEY:
        mess = "ANTHROPIC_API_KEY missing"
        logging.error(mess, exc_info=True)
        flash(mess)
        return mess, 0, "error"

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        # Determine max tokens based on model
        if "opus" in model:
            max_tokens = 4096
        elif "sonnet" in model:
            max_tokens = 4096
        else:  # haiku
            max_tokens = 4096

        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=SYSTEM_PROMPT + "\n\n" + custom_system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": prompt_blob,
                }
            ]
        )

        generated_text = response.content[0].text if response.content else ""
        if not generated_text:
            logging.warning("Anthropic API returned empty text")
            generated_text = "*Warning: Anthropic returned empty text*"

        finish_reason = response.stop_reason or "stop"
        if finish_reason != "end_turn":
            logging.warning(
                f"call_anthropic: non-nominal finish_reason='{finish_reason}' "
                f"model='{model}'"
            )

        input_rate, output_rate = get_rates(model)
        prompt_tokens = response.usage.input_tokens
        completion_tokens = response.usage.output_tokens
        cost = (
            prompt_tokens * input_rate
          + completion_tokens * output_rate
        ) / 1_000_000

        logging.info(
            f"Anthropic API used {prompt_tokens} prompt, "
            f"{completion_tokens} completion, "
            f"{prompt_tokens + completion_tokens} total tokens, "
            f"for a cost of ${cost}"
        )

        return generated_text, cost, finish_reason

    except Exception as e:
        mess = f"Error calling Anthropic API: {e}"
        logging.error(mess, exc_info=True)
        flash(mess)
        return mess, 0, "error"

Commit message: "Add call_anthropic() function for Anthropic API calls"


## Slice 3.4: Update get_rates() Function for Anthropic Models

File: app/routes.py

Add these cases to the existing get_rates() function:

# Add to the match statement in get_rates()
# Anthropic models
case "claude-3-opus-20240229":
    input_rate  = 15.00
    output_rate = 75.00
case "claude-3-sonnet-20240229":
    input_rate  = 3.00
    output_rate = 15.00
case "claude-3-haiku-20240307":
    input_rate  = 0.25
    output_rate = 1.25
case "claude-3-5-sonnet-20240620":
    input_rate  = 3.00
    output_rate = 15.00


## Slice 3.5: Add Anthropic Model Listing Route

File: app/routes.py

@app.route("/anthropic_modellst", methods=["GET"])
def anthropic_modellst() -> Any:
    """ Opens new page with a list of valid Anthropic model IDs. """
    results = get_anthropic_models()
    return render_template(
        "anthropic_modellst.html",
        results=results,
    )


## Slice 3.6: Update ui() Function to Handle Anthropic Platform

File: app/routes.py

In the ui() function, add Anthropic handling similar to the existing platform choices:

# Add after the digitalocean platform handling
if platform_choice == "anthropic":
    anthropic_models = get_anthropic_models()  # singleton cached
    if model not in anthropic_models:
        model = "claude-3-sonnet-20240229"

And update the template rendering to include anthropic_models:

return render_template(
    "ui.html",
    # ... existing parameters ...
    anthropic_models=anthropic_models,
    # ... rest of parameters ...
)


## Slice 3.7: Update UI Template for Anthropic Platform

File: app/templates/ui.html

Add Anthropic radio button and model dropdown:

<!-- Add to platform choice section -->
<input type="radio" name="platform_choice" value="anthropic" {% if platform_choice == 'anthropic' %}checked{% endif %}> Anthropic

<!-- Add to model dropdown section -->
{% elif platform_choice == "anthropic" %}
  <label>
    <a href="{{ url_for('anthropic_modellst') }}" target="_blank">Query Anthropic Models</a>
  </label>
  {% for m in anthropic_models %}
    <option value="{{ m }}"
      {% if m == model %}selected{% endif %}>
      {{ m }}
    </option>
  {% endfor %}


## slice 4 -- test and tshoot and fix and enhance code 
	Verify all three platforms work independently
	Test model switching between platforms
	Verify cost calculation for Anthropic models
