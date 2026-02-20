import logging
logging.basicConfig(level = logging.INFO,
                    filename = './code_assist.log',
                    filemode = 'a',
                    format = '%(asctime)s -%(levelname)s - %(message)s')
from flask import flash
import os
import shutil
import subprocess
import tempfile # Python standard library module for safely creating temporary files and directories


def clone_repo(repo_url: str) -> str:
    """
    Clone a git repository and return the local path.
    Supports GitHub and Codeberg URLs.
    Returns the path to the cloned repository or empty string on failure.
    """
    if not repo_url:
        return ""
    try:
        # Create a temporary directory for cloning
        temp_dir = tempfile.mkdtemp(prefix="code_assist_")
        result = subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, temp_dir],
            capture_output=True,
            text=True,
            check=True,
        )
        logging.info(f"Cloned repository {repo_url} to {temp_dir}")
        return temp_dir
    except Exception as e:
        logging.error(f"Failed to clone repository {repo_url}: {e}", exc_info=True)
        flash(f"Failed to clone repository: {repo_url}", "error")
        return ""


def cleanup_cloned_repo(repo_path: str):
    """
    Remove the cloned repository directory.
    """
    if repo_path and os.path.exists(repo_path):
        try:
            shutil.rmtree(repo_path)
            logging.info(f"Cleaned up cloned repository at {repo_path}")
        except Exception as e:
            logging.error(f"Failed to clean up cloned repository at {repo_path}: {e}", exc_info=True)
            flash(f"Failed to clean up cloned repository at {repo_path}", "error")

