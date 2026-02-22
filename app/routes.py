# pipenv install flask markdown-it-py pygments requests dotenv gradient
# sudo apt install ripgrep

import logging
logging.basicConfig(level = logging.INFO,
                    filename = './code_assist.log',
                    filemode = 'a',
                    format = '%(asctime)s -%(levelname)s - %(message)s')
from app import app
from collections import defaultdict
from dotenv import load_dotenv
from flask import render_template, request, redirect, url_for, flash  # session
from gradient import Gradient
from markdown_it import MarkdownIt
from pygments.formatters import HtmlFormatter
import datetime
import os
import requests
import subprocess

from app.repo_clone_management import clone_repo, cleanup_cloned_repo
from app.repo_scan import scan_repo_signals, scan_repo_signals, summarize_python_file
from app.repo_scan import summarize_html_file, summarize_css_file
from app.mistral_stuff import get_devstral_models, get_mismodlst, get_mismodcostlst
from app.digitalocean_stuff import get_digitalocean_models, get_domodlst, anthropic_dog_info


load_dotenv('../.env')
DEVSTRAL_API_KEY = os.environ.get("DEVSTRAL_API_KEY")
DEVSTRAL_API_URL = os.environ.get("DEVSTRAL_API_URL")
GRADIENT_MODEL_ACCESS_KEY = os.environ.get("GRADIENT_MODEL_ACCESS_KEY")

WRITE_PERMALOG = True
PERMALOG_FN = "./code_assist_perma.log"

LAST_PROMPT_FN = "./last_prompt.txt"

SYSTEM_PROMPT = """Do not introduce new layers, frameworks, or abstractions. 
Follow existing naming, error-handling, and logging conventions. 
If the task cannot be completed within the stated scope or allowed files, 
    stop and explain why before producing any code. 
Do not infer missing functionality or files. 
If something appears required but is absent, ask or stop. 
If assumptions are required, list them before proceeding. 
Output should be design notes and/or unified diff(s). 
Diffs need to be clean, readable, simple, and show inserts and deletes fully marked with no sub indents assumed.
No need to - and + existing code unchanged over itself; just deliver the changes. 
Use "exc_info=True" on all exception logging. 
Preferred output format: .md. 
Ignore functions with _old in their name except as reference to previous iterations of the code, these are not used.
Don't try to make code changes/fixes/improvements to code you don't see in full (guessing/imagining/assuming), 
    rather just answer explaining need to see X.py to make a patch for it.
Don't end with "...let me know and will proceed with implementation..."; 
    as a one-shot code assist there is no memory or history to continue and do a next step referencing this one.
"""

UNUSED_SYSTEM_PROMPT = """
(preferably with no explanations outside of code blocks, 
so as to keep things machine-checkable) 
"""

# Markdown + syntax highlighting
md = MarkdownIt("commonmark", {"html": True, "linkify": True})
formatter = HtmlFormatter()
PYGMENTS_CSS = formatter.get_style_defs(".highlight")


@app.route("/")
def index():
    return redirect(url_for("ui"))


@app.route("/ui", methods=["GET", "POST"])
def ui():
    """
    This is where everything happens for the "one-shot (repo-aware) code assist". 
    Three overall sections to code flow. First GET and POST init everything. 
    Then POSTs (both action "trigger" and "refresh_files") continue to 
    cycle form data to maintain state and 
    generates repo_files and file_tree, 
    then action "trigger" continues to use_ripgrep and 
    summarize_python_file or include files in full, 
    assemble blob, call_devstral, and then 
    finally return ui.html with a lot of content.
    Function is a little unwieldy.
    """
    model = "anthropic-claude-4.5-sonnet"  # just for first get
    platform_choice = "digitalocean"  # just for first get
    common_sense_filter = True  # default True
    local_dir = "."  # default .
    use_os_walk = False  # default False
    temperature = 0.2  # default 0.2
    use_ripgrep = True  # default True
    sum_unchecked = True  # default True
    save_responses = False  # default False
    auto_clean_temp = False  # default False
    custom_system_prompt = ""  # should start blank
    timeout = 45  # 45 or 90 # default 45
    file_tree = {}  # will need to refresh
    cost = 0.0  # should start zero
    git_repo_url = ""  # should start blank
    last_prompt = ""  # should start blank
    explicit_files = []  # should start blank
    cloned_repo_path = ""  # will be set when used
    prompt_blob = ""  # will be set on trigger post
    rendered_html = ""  # will be set on trigger post
    markdown_text = ""  # will be set on trigger post
    repo_files = []  # will marshal
    devstral_models = []  # will marshal
    dog_models = []  # will marshal
    repo_signals = ""  # will marshal
    if request.method == "POST":
        action = request.form.get("action", "trigger")
        logging.info(f'request.method == "POST"; action = "{action}"')
        model = request.form.get(
            "model",
            "devstral-latest"
        )
        custom_system_prompt = request.form.get("custom_system_prompt", "").strip()
        platform_choice = request.form.get("platform_choice", "mistral")
        if platform_choice == "mistral":
            if model not in devstral_models:
                devstral_models = get_devstral_models() # singleton cached
                model = "devstral-latest"
        if platform_choice == "digitalocean":
            dog_models = get_digitalocean_models() # singleton cached
            if model not in dog_models:
                model = "anthropic-claude-opus-4.6"
        local_dir = request.form.get("local_dir", "").strip()
        git_repo_url = request.form.get("git_repo_url", "").strip()
        auto_clean_temp = request.form.get("auto_clean_temp") == "1"
        use_os_walk = request.form.get("use_os_walk") == "1"
        use_ripgrep = request.form.get("use_ripgrep") == "1"
        save_responses = request.form.get("save_responses") == "1"
        try:
            timeout = int(request.form.get("timeout", timeout))
        except ValueError:
            timeout = 45
        # temperature (safe clamp)
        try:
            temperature = float(request.form.get("temperature", temperature))
            temperature = max(0.0, min(1.0, temperature))
        except ValueError:
            temperature = 0.2
        explicit_files = request.form.getlist("include_files")
        sum_unchecked = request.form.get("sum_unchecked") == "1"
        if git_repo_url: # Handle git_repo_url if provided
            cloned_repo_path = clone_repo(git_repo_url)
            if cloned_repo_path:
                local_dir = cloned_repo_path
            else:
                flash(f"Failed to clone repository from {git_repo_url} (fell back to local_dir)", "error")
                git_repo_url = ""  # Clear if cloning failed
        repo_files = get_repo_files(local_dir=local_dir or ".", use_os_walk=use_os_walk)
        common_sense_filter = request.form.get("common_sense_filter") == "1"
        if common_sense_filter:
            filtered_files = []
            for f in repo_files:
                if not any(f.endswith(ext) for ext in ['.toc', '.bin', '.env', '.log', '.ico', '.jpg', '.jpeg', '.png', '.gif', '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.zip', '.tar', '.gz', '.bz2', '.7z', '.rar']):
                    if not any(f.endswith(ext) for ext in ['Pipfile.lock', 'requirements.txt', 'package-lock.json', 'yarn.lock', 'node_modules']):
                        if not f.startswith('.') and '/.' not in f:
                            if not f.startswith("response_") and not f.endswith(".md"):
                                filtered_files.append(f)
            repo_files = filtered_files
        file_tree = build_file_tree(repo_files)
        if action == "trigger": # action == "refresh_files" runs the above only
            user_prompt = request.form.get("prompt", "").strip()
            if user_prompt:
                try:
                    with open(LAST_PROMPT_FN, 'w', encoding='utf-8') as _lpf:
                        _lpf.write(user_prompt)
                except IOError as e:
                    logging.error(f"Failed to write last_prompt file: {e}", exc_info=True)
                    flash(f"Failed to save last prompt: {e}")
                if use_ripgrep:
                    repo_signals = scan_repo_signals(local_dir)
                sel_files_blob = ""
                for f in repo_files:
                    path = os.path.join(local_dir, f)
                    if f in explicit_files:
                        # read full content
                        try:
                            with open(path, "r", encoding="utf-8") as fp:
                                content = fp.read()
                        except Exception:
                            content = f"# Could not read {f}"
                        sel_files_blob += f"### {f}\n```\n{content}\n```"
                    else:
                        if sum_unchecked: # summarize if sum_unchecked True otherwise skip file
                            extension = os.path.splitext(f)[1].lower() # extension with dot in lower case
                            match extension:
                                case ".py":
                                    summary = summarize_python_file(path)
                                case ".html":
                                    summary = summarize_html_file(path)
                                case ".css":
                                    summary = summarize_css_file(path)
                                case _:
                                    summary = "no summary"
                            sel_files_blob += f"### {f} (summary)\n```\n{summary}\n```"
                prompt_blob = "\n\n".join([
                    f"#### SOME REPOSITORY SIGNALS:",
                    repo_signals,
                    f"#### FULL AND FILE SUMMARIES:",
                    sel_files_blob,
                    f"#### USER QUERY/PROMPT:",
                    user_prompt,
                ])
                if platform_choice == "mistral":
                    # Invoke! Devstral API
                    markdown_text, cost = call_devstral(
                        prompt_blob=prompt_blob,
                        custom_system_prompt=custom_system_prompt,
                        model=model,
                        temperature=temperature,
                        timeout=timeout,
                    )
                else:
                    # Invoke! DigitalOcean Gradient API
                    markdown_text, cost = call_dog(
                        prompt_blob=prompt_blob,
                        custom_system_prompt=custom_system_prompt,
                        model=model,
                        temperature=temperature,
                        timeout=timeout,
                    )
                # Render Markdown for UI
                rendered_html = md.render(markdown_text)
                logging.info(f'got post and generated rendered_html')
                # Save response if checkbox is checked
                if save_responses and markdown_text:
                    os.makedirs('responses', exist_ok=True)
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"responses/response_{timestamp}.md"
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(user_prompt)
                        f.write(f"\n\n++++++++++++++++\n\n")
                        f.write(markdown_text)
                if WRITE_PERMALOG:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    try:
                        with open(PERMALOG_FN, 'a', encoding='utf-8') as f:
                            f.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n")
                            f.write(f"Cost: ${cost:.7f}, Model: {model}, Temperature: {temperature}\n")
                            f.write(f"Prompt: {user_prompt[:46]}{'...' if len(user_prompt) > 46 else ''}\n")
                            f.write(f"Response: {markdown_text[:46]}{'...' if len(markdown_text) > 46 else ''}\n")
                            f.write("-" * 80 + "\n")
                    except IOError as e:
                        mess = f"Failed to write to permalog: {e}"
                        logging.error(mess, exc_info=True)
                        flash(mess)
                    except Exception as e:
                        mess = f"Unexpected error writing to permalog: {e}"
                        logging.error(mess, exc_info=True)
                        flash(mess)
            else:
                rendered_html = "<em>No prompt provided.</em>"
        elif action == "refill_last_prompt":
            try:
                with open(LAST_PROMPT_FN, 'r', encoding='utf-8') as _lpf:
                    last_prompt = _lpf.read()
            except FileNotFoundError:
                last_prompt = ''
            except IOError as e:
                logging.error(f"Failed to read last_prompt file: {e}", exc_info=True)
                flash(f"Failed to read last prompt: {e}")
                last_prompt = ''
    if cloned_repo_path: # Clean up cloned repository if we used one
        if not auto_clean_temp:
            cleanup_cloned_repo(cloned_repo_path)
        else:
            flash(f"Note: Do clean up {cloned_repo_path} later when done.")
    logging.info(f"Static .css URL test: {url_for('static', filename='css/app.css')}")
    return render_template(
        "ui.html",
        rendered_html=rendered_html,
        cost=cost,
        pygments_css=PYGMENTS_CSS,
        sys_prompt_text=SYSTEM_PROMPT,
        prompt_blob=prompt_blob,
        repo_files=repo_files,
        file_tree=file_tree,
        selected_files=explicit_files,
        model=model,
        common_sense_filter=common_sense_filter,
        devstral_models=devstral_models,
        dog_models=dog_models,
        local_dir=local_dir,
        git_repo_url=git_repo_url,
        timeout=timeout,
        use_os_walk=use_os_walk,
        temperature=temperature,
        use_ripgrep=use_ripgrep,
        auto_clean_temp=auto_clean_temp,
        sum_unchecked=sum_unchecked,
        save_responses=save_responses,
        last_prompt=last_prompt,
        custom_system_prompt=custom_system_prompt,
        platform_choice=platform_choice,
    )


@app.route("/open_in_editor", methods=["POST"])
def open_in_editor():
    """
    Extracts code from rendered_html, saves it to a temporary file, and opens it in the default editor.
    """
    # Get the rendered_html from the form data
    rendered_html = request.form.get("rendered_html", "")
    # Extract the code from the rendered_html (assuming it's the last code block)
    code = ""
    if rendered_html:
        parts = rendered_html.split("```")
        if len(parts) >= 3:
            code = parts[-2].strip()
    # Prepend the header
    code = "# GENERATED BY DEVSTRAL – REVIEW BEFORE COMMIT\n\n" + code
    # Save to a temporary file
    output_path = request.form.get('filename', 'output.md')
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(code)
    except Exception as e:
        logging.error(f"Failed to write to file {output_path}: {e}")
        return f"*Error: Failed to write to file {output_path}: {e}*", 400

    # Open in the default editor
    try:
        if os.name == 'nt':  # Windows
            subprocess.Popen(['notepad.exe', output_path])
        elif os.name == 'posix':  # Linux or macOS
            subprocess.Popen(['xdg-open', output_path])
        else:
            logging.error("Unsupported operating system")
            flash("Unsupported operating system", "error")
            ### return "*Error: Unsupported operating system*", 400
    except Exception as e:
        logging.error(f"Failed to open file {output_path} in editor: {e}")
        flash(f"Failed to open file {output_path} in editor: {e}", "error")
        ### return f"*Error: Failed to open file {output_path} in editor: {e}*", 400

    return redirect(url_for("ui"))


def get_rates(model):
    """
    $ per 1M Input tokens
    $ per 1M Output tokens
    """
    match model:
        # Haiku
        case "anthropic-claude-3.5-haiku":
            input_rate  =  0.80
            output_rate =  4.00
        case "anthropic-claude-4.5-haiku", "anthropic-claude-haiku-4.5":
            input_rate  =  1.00
            output_rate =  5.00
        # Sonnet
        case "anthropic-claude-3.5-sonnet":
            input_rate  =  3.00
            output_rate = 15.00
        case "anthropic-claude-3.7-sonnet":
            input_rate  =  3.00
            output_rate = 15.00
        case "anthropic-claude-sonnet-4":
            input_rate  =  3.00
            output_rate = 15.00
        case "anthropic-claude-4.5-sonnet":
            input_rate  =  3.00
            output_rate = 15.00
        case "anthropic-claude-4.6-sonnet":
            input_rate  =  3.00
            output_rate = 15.00
        # Opus
        case "anthropic-claude-3-opus":
            input_rate  = 15.00
            output_rate = 75.00
        case "anthropic-claude-opus-4":
            input_rate  = 15.00
            output_rate = 75.00
        case "anthropic-claude-4.1-opus":
            input_rate  = 15.00
            output_rate = 75.00
        case "anthropic-claude-opus-4.5":
            input_rate  =  5.00
            output_rate = 25.00
        case "anthropic-claude-opus-4.6":
            input_rate  =  5.00  # DO billing console shows @ $0.005/thousand
            output_rate = 25.00  # DO billing console shows @ $0.025/thousand
        # devstral
        case "devstral-small-2507":
            input_rate  =  0.10
            output_rate =  0.30
        case "devstral-medium-2507":
            input_rate  =  0.40
            output_rate =  2.00
        case "devstral-2512":
            input_rate  =  0.40
            output_rate =  2.00
        case "devstral-medium-latest":
            input_rate  =  0.40
            output_rate =  2.00
        case "devstral-latest":
            input_rate  =  0.40
            output_rate =  2.00
        case "labs-devstral-small-2512":
            input_rate  =  0.10
            output_rate =  0.30
        case "devstral-small-latest":
            input_rate  =  0.10
            output_rate =  0.30
    return input_rate, output_rate


def call_dog(
    prompt_blob: str,
    custom_system_prompt: str,
    model: str = "anthropic-claude-opus-4.6",
    temperature: float = 0.2,
    timeout: int = 45,
) -> tuple[str, float]:
    """
    Call DigitalOcean Gradient API with given prompt, model, and temperature.
    Returns a tuple:
      - generated_text (Markdown-ready str)
      - cost (float)
    """
    if not GRADIENT_MODEL_ACCESS_KEY:
        mess = "GRADIENT_MODEL_ACCESS_KEY missing"
        logging.error(mess, exc_info=True)
        flash(mess)
        return mess, 0

    # Determine context window based on model
    # All current Anthropic Claude models via DigitalOcean Gradient support 200k tokens
    if model.startswith("anthropic-claude"):
        max_context = 200_000
    else:
        # Fallback for unknown models
        max_context = 100_000
        logging.warning(f"Unknown model '{model}', using fallback context window of {max_context}")

    try:
        client = Gradient(access_token=GRADIENT_MODEL_ACCESS_KEY)
        # Rough token estimate
        prompt_tokens_est = len(prompt_blob) // 4
        remaining = max_context - prompt_tokens_est - 512
        desired_max = min(4096, max(512, remaining))
        
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT + "\n\n" + custom_system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt_blob,
                },
            ],
            model=model,
            max_tokens=desired_max,
            temperature=temperature,
        )
        generated_text = response.choices[0].message.content
        if not generated_text:
            logging.warning("DigitalOcean Gradient API returned empty text")
            generated_text = "*Warning: DigitalOcean Gradient returned empty text*"
        input_rate, output_rate = get_rates(model)
        usage = response.usage
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        # total_tokens = response.usage.total_tokens
        cost = (
            prompt_tokens * input_rate
          + completion_tokens * output_rate
        ) / 1_000_000
        logging.info(
            f"DigitalOcean Gradient API used {prompt_tokens} prompt, "
            f"{completion_tokens} completion, "
            f"{prompt_tokens + completion_tokens} total tokens, "
            f"for a cost of ${cost}"
        )
        return generated_text, cost
    except Exception as e:
        mess = f"Error calling DigitalOcean Gradient API: {e}"
        logging.error(mess, exc_info=True)
        flash(mess)
        return mess, 0


def call_devstral(
    prompt_blob: str,
    custom_system_prompt: str,
    model: str = "devstral-latest",
    temperature: float = 0.2,
    timeout: int = 45,
) -> tuple[str, float]:
    """
    Call Devstral API with given prompt, model, and temperature.
    Returns a tuple:
      - generated_text (Markdown-ready str)
      - cost (float)
    """
    if not DEVSTRAL_API_KEY:
        mess = f"DEVSTRAL_API_KEY missing"
        logging.error(mess)
        flash(mess)
        return mess, 0
    # rough token estimate
    prompt_tokens_est = len(prompt_blob) // 4  
    if "small" in model.lower(): ### need to dial in numbers here better, also medium?
        max_context = 256 * 1024
    else:
        max_context = 252 * 1024
    remaining = max_context - prompt_tokens_est - 512
    desired_max = min(4096, max(512, remaining)) ### need to check these numbers more! (top was 16000)
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": desired_max,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT + "\n\n" + custom_system_prompt + "\n\n",
            },
            {
                "role": "user",
                "content": prompt_blob,
            },
        ],
    }
    headers = {
        "Authorization": f"Bearer {DEVSTRAL_API_KEY}",
        "Content-Type": "application/json",
    }
    match timeout:
        case 45:
            timeout_values = (6, 39)
        case 90:
            timeout_values = (13, 77)
        case _:
            timeout_values = (5, 30)
    try:
        response = requests.post(
            DEVSTRAL_API_URL,
            json=payload,
            headers=headers,
            timeout=timeout_values, # connect_timeout, read_timeout
        )
        response.raise_for_status() # Raise HTTPError if status not 2xx
        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            return "*Warning: No choices returned by Devstral*", 0
        generated_text = choices[0]["message"]["content"]
        if not generated_text:
            logging.warning("Devstral API returned empty text")
            generated_text = f"*Warning: Devstral returned empty text*"
        # Extract token usage if available
        input_rate, output_rate = get_rates(model)
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", prompt_tokens_est)
        completion_tokens = usage.get("completion_tokens", 0)
        cost = (
            prompt_tokens * input_rate
          + completion_tokens * output_rate
        ) / 1_000_000
        logging.info(
            f"Devstral API used {prompt_tokens} prompt, "
            f"{completion_tokens} completion, "
            f"{prompt_tokens + completion_tokens} total tokens, "
            f"for a cost of ${cost}"
        )
        return generated_text, cost
    except requests.HTTPError:
        mess = f"Devstral API error response\n{response.text}"
        logging.error(mess)
        flash(mess)
        return mess, 0
    except Exception as e:
        mess = f"Error calling Devstral API {e}"
        logging.error(mess, exc_info=True)
        flash(mess)
        return mess, 0


def get_repo_files(local_dir=".", use_os_walk=False):
    # Try git first unless explicitly told to use os.walk
    if not use_os_walk:
        try:
            result = subprocess.run(
                ["git", "ls-files"],
                cwd=local_dir,
                capture_output=True,
                text=True,
                check=True,
            )
            files = result.stdout.splitlines()
            if files:
                return files
        except Exception:
            pass

    # Fallback or forced os.walk
    try:
        import os
        files = []
        for root, dirs, filenames in os.walk(local_dir):
            for f in filenames:
                full_path = os.path.join(root, f)
                files.append(os.path.relpath(full_path, local_dir))
        return files
    except Exception:
        return []


def build_file_tree(paths):
    """Convert flat list of paths into nested dict for folder tree."""
    tree = lambda: defaultdict(tree)
    root = tree()
    for path in paths:
        parts = path.split("/")
        current = root
        for part in parts[:-1]:
            current = current[part]
        current[parts[-1]] = None  # files are leaves
    return root


@app.route("/mismodlst", methods=["GET"])
def mismodlst():
    """ Opens new page with a list of valid Mistral model IDs available to this API key. """
    results = get_mismodlst()
    return render_template(
        "mismodlst.html",
        results=results,
    )


@app.route("/domodlst", methods=["GET"])
def domodlst():
    """ Opens new page with a list of valid DigitalOcean Gradient model IDs available. """
    results = get_domodlst()
    return render_template(
        "domodlst.html",
        results=results,
    )


@app.route("/dog_info", methods=["GET"])
def dog_info():
    """ Opens new page with information about Anthropic's model lineup. """
    return render_template(
        "dog_info.html",
        info_text=anthropic_dog_info,
    )


@app.route("/mismodcostlst", methods=["GET"])
def mismodcostlst():
    """ Opens new page with a list of valid Mistral model IDs and costs available to this API key. """
    results = get_mismodcostlst()
    return render_template(
        "mismodlst.html",
        results=results,
    )


@app.route("/grep_py", methods=["GET"])
def grep_py():
    """
    Perform grep search and display results.
    """
    search_string = request.args.get("q", "").strip()
    local_dir = request.args.get("local_dir", ".")

    if not search_string:
        flash("No search string provided", "error")
        return redirect(url_for("ui"))

    try:
        result = subprocess.run(
            ["grep", "-rn", "--include=*.py", "--include=*.html", "--include=*.js", "--include=*.css", search_string],
            cwd=local_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        return render_template(
            "grep_results.html",
            search_string=search_string,
            pygments_css=PYGMENTS_CSS,
            results=result.stdout,
            error=result.stderr,
        )
    except Exception as e:
        flash(f"Error performing grep: {e}", "error")
        return redirect(url_for("ui"))

