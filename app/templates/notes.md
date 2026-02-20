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
