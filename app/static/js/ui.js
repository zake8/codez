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
  const selectedFile = encodeURIComponent(select.value);
  const button = document.getElementById('mypy_run_btn');
  const localDir = encodeURIComponent(button.getAttribute('data-local-dir') || '.');
  if (!select.value) {
    alert('Please select a Python file first.');
    return;
  }
  console.log(`Opening: /mypy_file?f=${selectedFile}&local_dir=${localDir}`);
  window.open(`/mypy_file?f=${selectedFile}&local_dir=${localDir}`, '_blank');
}


document.addEventListener('DOMContentLoaded', function() {
  const form = document.querySelector('form[method="POST"]');
  const spinner = document.getElementById('spinner');
  // After a trigger POST, scroll to the response section if it has content
  var responseAnchor = document.getElementById('response');
  var chatMessage = document.querySelector('.output .chat-message');
  if (responseAnchor && chatMessage && chatMessage.textContent.trim().length > 0) {
    responseAnchor.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
  if (form && spinner) {
    form.addEventListener('submit', function() {
      spinner.style.display = 'inline';
    });
  }
});
