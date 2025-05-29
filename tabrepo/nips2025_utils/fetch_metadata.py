from __future__ import annotations

from pathlib import Path

import pandas as pd

from autogluon.common.loaders import load_pd
import tabrepo


def load_task_metadata(paper: bool = True) -> pd.DataFrame:
    tabrepo_root = str(Path(tabrepo.__file__).parent.parent)
    if paper:
        path = f"{tabrepo_root}/tabflow/metadata/task_metadata_tabarena51.csv"
    else:
        path = f"{tabrepo_root}/tabflow/metadata/task_metadata_tabarena61.csv"
    task_metadata = load_pd.load(path=path)
    task_metadata["dataset"] = task_metadata["name"]
    return task_metadata
