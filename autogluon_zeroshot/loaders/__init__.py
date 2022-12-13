from pathlib import Path

from ._configs import load_configs
from ._results import load_results, combine_results_with_score_val


class Paths:
    data_root: Path = Path(__file__).parent.parent.parent / 'data'
    results_root: Path = data_root / 'results'
    all_v3_results_root: Path = results_root / "all_v3"
    bagged_results_root: Path = results_root / "bagged"
