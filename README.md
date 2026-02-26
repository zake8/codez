

# Repo-level Code Assistant

A Flask web app for one-shot, repo-aware code assistance.
Point it at a local directory or a GitHub/Codeberg repository URL,
and it assembles a prompt blob from ripgrep signals and file
summaries, then sends it to either the Mistral (Devstral) API
or the DigitalOcean Gradient (Anthropic Claude) API.


## Features

- **Repo-aware context** — ripgrep scans the target repo for
  symbols, call sites, and conventions and prepends the results
  to every prompt.
- **Selective file inclusion** — browse a file tree in the UI,
  check files to include in full, or let the app summarize
  unchecked `.py`, `.html`, and `.css` files automatically.
- **Two AI backends**
  - **Mistral / Devstral** via the Mistral API.
  - **Anthropic Claude** via DigitalOcean Gradient.
- **Git repo cloning** — paste a GitHub or Codeberg URL and the
  app clones it to a temp directory, scans it, and optionally
  cleans up afterward.
- **Cost tracking** — per-request token usage and estimated cost
  are shown in the response header and written to a permanent log.
- **Response saving** — optionally write each response to
  `responses/response_{timestamp}.md`.
- **Grep search** — search `.py`, `.html`, `.js`, and `.css`
  files in the target repo from within the UI.
- **Open in editor** — send the last response to a local text
  editor (Linux: `xdg-open`, Windows: `notepad.exe`).
- **Markdown rendering** — responses are rendered with
  `markdown-it-py` and syntax-highlighted with Pygments.


## Installation

1. Clone the repository:

`git clone https://github.com/your-org/your-repo.git`

`cd your-repo`

2. Install system dependencies:

`sudo apt install ripgrep`

3. Install Python dependencies:

`pipenv install`

4. Create a `.env` file (can rename `sample.env`).


## Running

Start the server:

`pipenv run python app.py`

Then open `http://localhost:5000` in a browser.

For development with auto-reload:

`pipenv run flask --app app run --debug`



## Logs

- `code_assist.log` — application log
- `code_assist_perma.log` — permanent one-line-per-request log
- Optional `response_*.md` files in `/responses` dir

