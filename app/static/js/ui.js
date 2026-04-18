// ── Directory browser ────────────────────────────────────────────────────────

function attrJson(val) {
  // JSON.stringify produces double-quoted strings; when embedded inside an
  // HTML attribute that is itself double-quoted the inner quotes break parsing.
  // Escape them to &quot; so the attribute stays well-formed.
  return JSON.stringify(val).replace(/"/g, '&quot;');
}

function openDirBrowser() {
  const localDirInput = document.getElementById('local_dir_input');
  const startPath = localDirInput.value.trim() || '/';
  const overlay = document.getElementById('dir-browser-overlay');
  overlay.style.display = 'flex';
  loadDirBrowser(startPath);
}

function closeDirBrowser() {
  const overlay = document.getElementById('dir-browser-overlay');
  overlay.style.display = 'none';
}

function loadDirBrowser(path) {
  const statusEl = document.getElementById('dir-browser-status');
  const listEl   = document.getElementById('dir-browser-list');
  const pathEl   = document.getElementById('dir-browser-path');

  statusEl.textContent = 'Loading…';
  listEl.innerHTML = '';

  fetch('/browse_dir?path=' + encodeURIComponent(path))
    .then(function(resp) { return resp.json(); })
    .then(function(data) {
      if (data.error) {
        statusEl.textContent = 'Error: ' + data.error;
        return;
      }
      statusEl.textContent = '';
      pathEl.textContent = data.path;

      // "Select this directory" button at top
      const selectBtn = document.createElement('li');
      selectBtn.innerHTML =
        '<button type="button" class="dir-browser-select-btn" ' +
        'onclick="selectDir(' + attrJson(data.path) + ')">✔ Select: ' +
        escHtml(data.path) + '</button>';
      listEl.appendChild(selectBtn);

      // Parent directory link
      if (data.parent !== null) {
        const parentLi = document.createElement('li');
        parentLi.innerHTML =
          '<span class="dir-browser-entry dir-browser-dir" ' +
          'onclick="loadDirBrowser(' + attrJson(data.parent) + ')">⬆ ..</span>';
        listEl.appendChild(parentLi);
      }

      // Entries — directories first (server already sorted alpha, dirs flagged)
      data.entries.forEach(function(entry) {
        const li = document.createElement('li');
        if (entry.is_dir) {
          li.innerHTML =
            '<span class="dir-browser-entry dir-browser-dir" ' +
            'onclick="loadDirBrowser(' + attrJson(entry.path) + ')">📁 ' +
            escHtml(entry.name) + '</span>';
        } else {
          li.innerHTML =
            '<span class="dir-browser-entry dir-browser-file">📄 ' +
            escHtml(entry.name) + '</span>';
        }
        listEl.appendChild(li);
      });
    })
    .catch(function(err) {
      statusEl.textContent = 'Fetch error: ' + err;
    });
}

function selectDir(path) {
  document.getElementById('local_dir_input').value = path;
  closeDirBrowser();
}

function escHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ── End directory browser ────────────────────────────────────────────────────

function grepPy() {
  const searchValue = encodeURIComponent(document.getElementById('grep_search').value);
  const button = event.target;
  const localDir = encodeURIComponent(button.getAttribute('data-local-dir') || '.');
  console.log(`Opening: /grep_py?q=${searchValue}&local_dir=${localDir}`);
  window.open(`/grep_py?q=${searchValue}&local_dir=${localDir}`, '_blank');
}


function pyHelp() {
  const searchValue = encodeURIComponent(document.getElementById('pyhelp_search').value);
  const button = event.target;
  const localDir = encodeURIComponent(button.getAttribute('data-local-dir') || '.');
  console.log(`Opening: /pyhelp?q=${searchValue}&local_dir=${localDir}`);
  window.open(`/pyhelp?q=${searchValue}&local_dir=${localDir}`, '_blank');
}


function mypyFile() {
  const select = document.getElementById('mypy_file_select');
  const button = document.getElementById('mypy_run_btn');
  const localDir = encodeURIComponent(button.getAttribute('data-local-dir') || '.');
  if (!select.value) {
    alert('Please select a .py file or "." for the whole repo.');
    return;
  }
  const selectedFile = encodeURIComponent(select.value);
  const flagBoxes = document.querySelectorAll('.mypy-flag:checked');
  const flags = Array.from(flagBoxes).map(cb => encodeURIComponent(cb.value));
  let url = `/mypy_file?f=${selectedFile}&local_dir=${localDir}`;
  for (const flag of flags) {
    url += `&mypy_flags=${flag}`;
  }
  console.log(`Opening: ${url}`);
  window.open(url, '_blank');
}


document.addEventListener('DOMContentLoaded', function() {
  const spinner = document.getElementById('spinner');
  const responseSection = document.getElementById('response');
  const chatMessage = document.querySelector('.output .chat-message');

  // Scroll to response if it has content (i.e. we just got a POST response back)
  if (responseSection && chatMessage && chatMessage.textContent.trim().length > 0) {
    responseSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  // Show spinner on any submit button click that is an "action=trigger" submit
  document.querySelectorAll('button[value="trigger"]').forEach(function(btn) {
    btn.addEventListener('click', function() {
      if (spinner) spinner.style.display = 'inline';
    });
  });
});
