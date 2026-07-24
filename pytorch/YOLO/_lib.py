"""Put the parent pytorch/ dir on sys.path so YOLO scripts can import the
flat toolkit modules (data, trainer, export, netviz, viewer) directly.

Import this first:  import _lib  # noqa: F401  (before any toolkit import)
"""

import sys
from pathlib import Path

_PARENT = str(Path(__file__).resolve().parent.parent)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
