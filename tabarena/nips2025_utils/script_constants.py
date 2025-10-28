from __future__ import annotations

from pathlib import Path

import tabarena

tabarena_root = str(Path(tabarena.__file__).parent.parent)
tabarena_data_root = Path(tabarena_root).parent.parent / "data"
