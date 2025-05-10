from __future__ import annotations

from pathlib import Path

import tabrepo

tabarena_root = str(Path(tabrepo.__file__).parent.parent)
tabarena_data_root = Path(tabarena_root).parent.parent / "data"
