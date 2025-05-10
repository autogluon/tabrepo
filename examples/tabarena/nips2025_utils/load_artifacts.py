from __future__ import annotations

from autogluon.common.loaders import load_pkl
from tabrepo.benchmark.result import ExperimentResults
from tabrepo.utils.parallel_for import parallel_for
from pickle import UnpicklingError
import time


def load_and_check_if_valid(path) -> bool:
    try:
        data = load_pkl.load(path)
        return True
    except UnpicklingError as err:
        return False
    except EOFError as err:
        return False


def load_and_align(path):
    data = load_pkl.load(path)
    data_aligned = ExperimentResults._align_result_input_format(data)
    return data_aligned


def load_all_artifacts(file_paths: list[str], engine: str = "sequential"):
    file_paths_lst = []
    for file_path in file_paths:
        file_paths_lst.append(
            {
                "path": str(file_path),
            }
        )

    ts = time.time()
    results_lst = parallel_for(
        f=load_and_align,
        inputs=file_paths_lst,
        engine=engine,
    )
    te = time.time()
    print(f"{te - ts:.2f}s\t{engine}")
    return results_lst
