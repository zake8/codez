"""
Script contains (helper) functions which 
do all the ripgrep stuff. 
Need to install tho: 
sudo apt install ripgrep
"""

import logging
logging.basicConfig(level = logging.INFO,
                    filename = './code_assist.log',
                    filemode = 'a',
                    format = '%(asctime)s -%(levelname)s - %(message)s')
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Any
import ast
import os
import re
import shutil
import subprocess


def _rg(pattern: str, cwd: str, max_lines: int = 200) -> list[str]:
    """
    Run ripgrep and return lines (bounded).
    """
    try:
        result = subprocess.run(
            ["rg", "--line-number", pattern],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,  # rg returns nonzero if no matches
        )
        lines = result.stdout.splitlines()
        return lines[:max_lines]
    except Exception as e:
        logging.warning(f"Error running rg: {e}", exc_info=True)
        return []


def _summarize_symbols(cwd: str) -> Tuple[Dict[str, int], Dict[str, list[str]]]:
    patterns = {
        "python_functions": r"^\s*def\s+\w+",
        "python_classes": r"^\s*class\s+\w+",
        "flask_routes": r"@(app|bp)\.route",
        "cli_entries": r"if\s+__name__\s*==\s*['\"]__main__['\"]",
    }
    counts = {}
    examples = {}
    for name, pattern in patterns.items():
        lines = _rg(pattern, cwd)
        counts[name] = len(lines)
        examples[name] = lines[:5]
    return counts, examples


def _summarize_call_sites(cwd: str) -> Dict[str, int]:
    symbols = ["Auth", "Service", "Repository", "Controller"]
    hits: Dict[str, int] = defaultdict(int)
    for sym in symbols:
        lines = _rg(sym, cwd)
        hits[sym] += len(lines)

    return dict(hits)


def _summarize_conventions(cwd: str) -> Dict[str, Any]:
    conventions: Dict[str, Any] = {}
    # Error handling
    error_lines = (
        _rg(r"raise\s+\w+Error", cwd)
        + _rg(r"except\s+\w+Error", cwd)
    )
    conventions["error_handling"] = len(error_lines)
    # Logging
    log_lines = _rg(r"logger\.(debug|info|warning|error|exception)", cwd)
    levels: Counter[str] = Counter()
    for line in log_lines:
        m = re.search(r"logger\.(\w+)", line)
        if m:
            levels[m.group(1)] += 1
    conventions["logging"] = dict(levels)
    # Config
    env_lines = _rg(r"os\.environ|process\.env", cwd)
    conventions["env_config"] = len(env_lines)
    return conventions


def has_ripgrep() -> bool:
    return shutil.which("rg") is not None


def scan_repo_signals(cwd: str) -> str:
    """
    Returns a markdown blob for prompt context 
    with all kinds of info found via ripgrep.
    """
    if not cwd:
        cwd = "."
    if not has_ripgrep():
        mess = f"Ripgrep (rg) not found in PATH! Unable to gather data."
        logging.warning(mess)
        return mess
    symbol_counts, symbol_examples = _summarize_symbols(cwd)
    call_sites = _summarize_call_sites(cwd)
    conventions = _summarize_conventions(cwd)
    lines = []
    lines.append("## Repo Signals (ripgrep-based)")
    lines.append("")
    # Symbols
    lines.append("### Symbols & Entry Points")
    for k, v in symbol_counts.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    # Examples
    lines.append("### Symbol Examples")
    for k, examples in symbol_examples.items():
        if not examples:
            continue
        lines.append(f"**{k}:**")
        for ex in examples:
            lines.append(f"- `{ex}`")
        lines.append("")
    # Call sites
    lines.append("### Call-site Signals")
    for sym, count in call_sites.items():
        lines.append(f"- `{sym}` references: {count}")
    lines.append("")
    # Conventions
    lines.append("### Conventions")
    lines.append(f"- Error handling patterns: {conventions['error_handling']}")
    lines.append(f"- Env-based config usage: {conventions['env_config']}")
    if conventions["logging"]:
        lines.append("- Logging levels:")
        for level, count in conventions["logging"].items():
            lines.append(f"  - {level}: {count}")
    return "\n".join(lines)


def summarize_python_file(filepath: str) -> str:
    """
    Summarizes a Python file by extracting:
    - Module docstring
    - Top-level imports
    - Top-level functions with decorators and docstrings
    - Top-level classes with methods
    - Top-level assignments/constants
    Returns a markdown-ready summary string.
    """
    try:
        # Get file stats
        file_stats = os.stat(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()
        line_count = len(source.splitlines())
    except Exception as e:
        return f"*Could not read file {filepath}: {e}*"
    try:
        if filepath.endswith("Pipfile"): # list unparseables here
            return f""
        tree = ast.parse(source, filename=filepath) # source code string cannot contain null bytes
    except SyntaxError as e:
        return f"*Could not parse file {filepath}: {e}*"
    summary_lines: List[str] = []
    # File metadata
    summary_lines.append(f"**File metadata:**")
    summary_lines.append(f"- Line count: {line_count}")
    summary_lines.append(f"- File size: {file_stats.st_size} bytes")
    summary_lines.append("")
    # Module docstring
    module_doc = ast.get_docstring(tree)
    if module_doc:
        summary_lines.append("**Module docstring:**")
        summary_lines.append(f"```\n{module_doc.strip()}\n```")
    # Top-level imports
    imports = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                imports.extend([alias.name for alias in node.names])
            else:
                module = node.module if node.module else ""
                imports.extend([f"{module}.{alias.name}" for alias in node.names])
    if imports:
        summary_lines.append("**Top-level imports:**")
        summary_lines.append(f"```\n{', '.join(sorted(imports))}\n```")
    # Top-level functions
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if functions:
        summary_lines.append("**Top-level functions:**")
        for fn in functions:
            decorators = [ast.unparse(d) for d in fn.decorator_list] if fn.decorator_list else []
            summary_lines.append(f"- `{fn.name}`")
            if decorators:
                summary_lines.append(f"  - Decorators: {', '.join(decorators)}")
            fn_doc = ast.get_docstring(fn)
            if fn_doc:
                summary_lines.append(f"  - Docstring:\n    ```\n    {fn_doc.strip()}\n    ```")
    # Top-level classes
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    if classes:
        summary_lines.append("**Top-level classes:**")
        for cls in classes:
            summary_lines.append(f"- `{cls.name}`")
            cls_doc = ast.get_docstring(cls)
            if cls_doc:
                summary_lines.append(f"  - Docstring:\n    ```\n    {cls_doc.strip()}\n    ```")
            # Methods
            methods = [m for m in cls.body if isinstance(m, ast.FunctionDef)]
            if methods:
                summary_lines.append("  - Methods:")
                for m in methods:
                    decorators = [ast.unparse(d) for d in m.decorator_list] if m.decorator_list else []
                    method_line = f"    - `{m.name}`"
                    if decorators:
                        method_line += f" (Decorators: {', '.join(decorators)})"
                    summary_lines.append(method_line)
    # Top-level constants / global variables
    constants = [n for n in tree.body if isinstance(n, ast.Assign)]
    if constants:
        summary_lines.append("**Top-level assignments/constants:**")
        for c in constants:
            targets = [ast.unparse(t) for t in c.targets]
            summary_lines.append(f"- `{', '.join(targets)}`")
    return "\n\n".join(summary_lines)


def summarize_html_file(filepath: str) -> str:
    """
    Summarizes an HTML file by extracting:
    - File metadata
    - Forms (with their action endpoints and methods)
    - Key frontend components (divs with IDs/classes, scripts)
    - Links and endpoints
    - Form fields
    - Meta tags
    - HTML5 elements
    - Data attributes
    - JavaScript event handlers
    Returns a markdown-ready summary string.
    """
    try:
        # Get file stats
        file_stats = os.stat(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()
        line_count = len(source.splitlines())
    except Exception as e:
        return f"*Could not read file {filepath}: {e}*"
    summary_lines: List[str] = []
    # File metadata
    summary_lines.append(f"**File metadata:**")
    summary_lines.append(f"- Line count: {line_count}")
    summary_lines.append(f"- File size: {file_stats.st_size} bytes")
    summary_lines.append("")
    # Forms
    form_pattern = r'<form\s+[^>]*action=["\']([^"\']+)["\'][^>]*method=["\']([^"\']+)["\']'
    forms = re.findall(form_pattern, source, re.IGNORECASE)
    if forms:
        summary_lines.append("**Forms:**")
        for action, method in forms:
            summary_lines.append(f"- Endpoint: `{action}` (Method: `{method}`)")
    # Form fields
    input_pattern = r'<input\s+[^>]*name=["\']([^"\']+)["\']'
    inputs = re.findall(input_pattern, source, re.IGNORECASE)
    if inputs:
        summary_lines.append("**Form fields:**")
        for input_name in inputs:
            summary_lines.append(f"- Input field: `{input_name}`")
    # Key frontend components (divs with IDs/classes)
    div_pattern = r'<div\s+[^>]*(id|class)=["\']([^"\']+)["\']'
    divs = re.findall(div_pattern, source, re.IGNORECASE)
    if divs:
        summary_lines.append("**Key frontend components:**")
        for attr, value in divs:
            summary_lines.append(f"- `{attr}={value}`")
    # Scripts
    script_pattern = r'<script\s+[^>]*src=["\']([^"\']+)["\']'
    scripts = re.findall(script_pattern, source, re.IGNORECASE)
    if scripts:
        summary_lines.append("**External scripts:**")
        for script in scripts:
            summary_lines.append(f"- `{script}`")
    # Inline scripts
    inline_script_pattern = r'<script[^>]*>(.*?)</script>'
    inline_scripts = re.findall(inline_script_pattern, source, re.IGNORECASE | re.DOTALL)
    if inline_scripts:
        summary_lines.append("**Inline scripts:**")
        for script in inline_scripts:
            script_content = script.strip()
            if script_content:
                summary_lines.append(f"- Script content: `{script_content[:50]}...`")
    # Links and endpoints
    link_pattern = r'<a\s+[^>]*href=["\']([^"\']+)["\']'
    links = re.findall(link_pattern, source, re.IGNORECASE)
    if links:
        summary_lines.append("**Links/endpoints:**")
        for link in links:
            summary_lines.append(f"- `{link}`")
    # Meta tags
    meta_pattern = r'<meta\s+[^>]*name=["\']([^"\']+)["\'][^>]*content=["\']([^"\']+)["\']'
    metas = re.findall(meta_pattern, source, re.IGNORECASE)
    if metas:
        summary_lines.append("**Meta tags:**")
        for name, content in metas:
            summary_lines.append(f"- `{name}: {content}`")
    # HTML5 elements
    html5_elements = ['header', 'footer', 'nav', 'section', 'article', 'aside', 'main']
    for element in html5_elements:
        element_pattern = rf'<{element}\b[^>]*>'
        if re.search(element_pattern, source, re.IGNORECASE):
            summary_lines.append(f"**HTML5 elements:**")
            summary_lines.append(f"- `{element}`")
            break
    # Data attributes
    data_pattern = r'data-[^=]+=["\'][^"\']+["\']'
    data_attrs = re.findall(data_pattern, source)
    if data_attrs:
        summary_lines.append("**Data attributes:**")
        for attr in data_attrs:
            summary_lines.append(f"- `{attr}`")
    # JavaScript event handlers
    event_pattern = r'on\w+\s*=\s*["\'][^"\']+["\']'
    events = re.findall(event_pattern, source)
    if events:
        summary_lines.append("**JavaScript event handlers:**")
        for event in events:
            summary_lines.append(f"- `{event}`")
    return "\n\n".join(summary_lines)


def summarize_css_file(filepath: str) -> str:
    """
    Summarizes a CSS file by extracting:
    - File metadata
    - CSS rules (selectors and their properties, including nested rules)
    - Media queries
    - Keyframes
    - Imports and variables
    - CSS custom properties usage
    - CSS animations and transitions
    - CSS Grid and Flexbox layouts
    - Responsive design patterns
    Returns a markdown-ready summary string.
    """
    try:
        # Get file stats
        file_stats = os.stat(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()
        line_count = len(source.splitlines())
    except Exception as e:
        return f"*Could not read file {filepath}: {e}*"
    summary_lines: List[str] = []
    # File metadata
    summary_lines.append(f"**File metadata:**")
    summary_lines.append(f"- Line count: {line_count}")
    summary_lines.append(f"- File size: {file_stats.st_size} bytes")
    summary_lines.append("")
    # CSS imports
    import_pattern = r'@import\s+["\']([^"\']+)["\']'
    imports = re.findall(import_pattern, source)
    if imports:
        summary_lines.append("**CSS imports:**")
        for imp in imports:
            summary_lines.append(f"- `{imp}`")
    # CSS variables
    var_pattern = r'--\w+\s*:\s*[^;]+;'
    variables = re.findall(var_pattern, source)
    if variables:
        summary_lines.append("**CSS variables:**")
        for var in set(variables):  # Remove duplicates
            summary_lines.append(f"- `{var.strip()}`")
    # CSS custom properties usage
    var_usage_pattern = r'var\(--\w+\)'
    var_usages = re.findall(var_usage_pattern, source)
    if var_usages:
        summary_lines.append("**CSS custom properties usage:**")
        for usage in set(var_usages):  # Remove duplicates
            summary_lines.append(f"- `{usage}`")
    # Media queries
    media_pattern = r'@media\s+([^{]+)\{[^}]*\}'
    media_queries = re.findall(media_pattern, source, re.DOTALL)
    if media_queries:
        summary_lines.append("**Media queries:**")
        for mq in media_queries:
            # Extract just the condition part for display
            condition = mq.split('{')[0].strip()
            summary_lines.append(f"- `{condition}`")
    # Keyframes
    keyframe_pattern = r'@keyframes\s+(\w+)\s*\{[^}]*\}'
    keyframes = re.findall(keyframe_pattern, source, re.DOTALL)
    if keyframes:
        summary_lines.append("**Keyframes:**")
        for kf in keyframes:
            summary_lines.append(f"- `{kf}`")
    # CSS animations and transitions
    animation_pattern = r'animation\s*:\s*[^;]+'
    animations = re.findall(animation_pattern, source)
    transition_pattern = r'transition\s*:\s*[^;]+'
    transitions = re.findall(transition_pattern, source)
    if animations or transitions:
        summary_lines.append("**CSS animations and transitions:**")
        for anim in animations:
            summary_lines.append(f"- Animation: `{anim.strip()}`")
        for trans in transitions:
            summary_lines.append(f"- Transition: `{trans.strip()}`")
    # CSS Grid and Flexbox layouts
    grid_pattern = r'display\s*:\s*grid'
    flexbox_pattern = r'display\s*:\s*flex'
    if re.search(grid_pattern, source):
        summary_lines.append("**CSS Grid layouts:**")
        summary_lines.append("- Grid layout detected")
    if re.search(flexbox_pattern, source):
        summary_lines.append("**CSS Flexbox layouts:**")
        summary_lines.append("- Flexbox layout detected")
    # CSS rules (selectors)
    rule_pattern = r'([^{]+)\{([^}]*)\}'
    rules = re.findall(rule_pattern, source, re.DOTALL)
    if rules:
        summary_lines.append("**CSS rules:**")
        for selector, properties in rules:
            selector = selector.strip()
            properties = properties.strip()
            summary_lines.append(f"- Selector: `{selector}`")
            if properties:
                # Count non-empty properties
                props = [p.strip() for p in properties.split(';') if p.strip()]
                prop_count = len(props)
                summary_lines.append(f"  - Properties: {prop_count}")
                # For nested rules, indicate complexity
                if '{' in properties and '}' in properties:
                    summary_lines.append(f"  - Contains nested rules")
    return "\n\n".join(summary_lines)

