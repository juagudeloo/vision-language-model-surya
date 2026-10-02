"""_paths.py — Shared path resolution for notebook_utils modules.

Resolved relative to this file, not the caller's cwd, so every module here
works the same regardless of which notebook imports it.
"""

import sys
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = NOTEBOOK_DIR.parent.parent
SURYA_ROOT = REPO_ROOT / "Surya"
DATA_DIR = NOTEBOOK_DIR / "data"

if str(SURYA_ROOT) not in sys.path:
    sys.path.append(str(SURYA_ROOT))
