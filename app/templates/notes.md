## issue is some text content comes thru as commands and the commands fail due to bad char in their paths

## in Prompt Blob viewer in ui.html

## code
    <!-- CONTEXT VIEWER -->
    <section class="context-viewer">
      <details>
        <summary>Prompt Blob</summary>
        <pre>{{ prompt_blob }}</pre>
        <pre>{{ prompt_blob|safe }}</pre>
        <pre>{{ prompt_blob|e }}</pre>
      </details>
    </section>

## what this is supposed to do
The `|e` filter (`|escape`) will HTML-entity-encode the content so 
`<script>`, `<link>`, and `{{ }}` are displayed as visible text 
rather than parsed as HTML/URLs by the browser.

## more
- explicitly HTML-escapes the string. All <, >, &, ", ' characters become their entity equivalents (&lt;, &gt;, etc.), so they display as literal text inside the <pre> block
- Flask's default auto-escape
- The content from summarize_html_file() (which returns strings containing HTML tags like <form, <div>, <script>) is the source of the display problems when |safe was used — those tags were being rendered as live HTML rather than displayed as text.

## +++++++++++++++++++++++++++++++++++++

## fixes:

The problem is not Jinja2 template injection — {{ prompt_blob|e }} correctly HTML-escapes the content. The actual failure mode is that prompt_blob contains literal {{ and }} sequences (from file summaries that include Jinja2 template source, i.e. ui.html itself), which Jinja2 tries to evaluate before the |e filter runs, causing template errors.

The fix: pass prompt_blob through Python's markupsafe.escape() in the route before handing it to the template, then render it with |safe. This way Jinja2 never sees the raw {{/}} — it only sees already-escaped HTML entities.

markup_escape(prompt_blob) runs in Python, converting {, }, <, >, &, ", ' to their HTML entities before Jinja2 ever touches the string as a template expression.

The result is a Markup object (a str subclass). Casting to str gives a plain string of safe HTML entities.

In the template, |safe tells Jinja2 "do not escape this again" — which is correct because it is already fully escaped. The <pre> block then displays the literal characters of the prompt blob as the user expects.

{{ prompt_blob|e }} failed because Jinja2 evaluates {{ prompt_blob }} first (hitting {{ inside the value and erroring), then applies |e. The Python-side escape sidesteps this entirely.
