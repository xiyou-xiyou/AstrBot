from __future__ import annotations

import sys
from pathlib import Path


PLUGINS_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[4]

for path in (PROJECT_ROOT, PLUGINS_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
