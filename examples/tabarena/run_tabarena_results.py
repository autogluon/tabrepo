from __future__ import annotations

from pathlib import Path

import pandas as pd

from tabrepo.nips2025_utils import load_results
from tabrepo.nips2025_utils.eval_all import evaluate_all


if __name__ == '__main__':
    # load the TabArena paper results
    df_results: pd.DataFrame = load_results()

    eval_save_path = Path("tabarena_paper") / "output"
    elo_bootstrap_rounds = 10  # 100 to reproduce the paper

    # regenerate figures and tables
    evaluate_all(df_results=df_results, eval_save_path=eval_save_path, elo_bootstrap_rounds=elo_bootstrap_rounds)
