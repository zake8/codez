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
