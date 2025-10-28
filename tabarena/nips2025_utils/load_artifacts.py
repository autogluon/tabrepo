from __future__ import annotations

from pathlib import Path

from autogluon.common.loaders import load_pkl
from tabarena.benchmark.result import AGBagResult, BaselineResult
from tabarena.utils.parallel_for import parallel_for


def load_and_align(path, convert_to_holdout: bool = False) -> BaselineResult:
    data: dict | BaselineResult = load_pkl.load(path)
    data_aligned = BaselineResult.from_dict(data)
    if convert_to_holdout:
        return result_to_holdout(result=data_aligned)
    return data_aligned


def load_all_artifacts(
    file_paths: list[str | Path],
    engine: str = "sequential",
    convert_to_holdout: bool = False,
    progress_bar: bool = True,
) -> list[BaselineResult]:
    file_paths_lst = []
    for file_path in file_paths:
        file_paths_lst.append(
            {
                "path": str(file_path),
                "convert_to_holdout": convert_to_holdout,
            }
        )

    results_lst: list[BaselineResult] = parallel_for(
        f=load_and_align,
        inputs=file_paths_lst,
        engine=engine,
        progress_bar=progress_bar,
        desc=f"Loading raw artifacts"
    )
    return results_lst


def result_to_holdout(result: BaselineResult) -> BaselineResult:
    assert isinstance(result, AGBagResult)
    result_holdout = result.bag_artifacts(as_baseline=False)
    if len(result_holdout) > 0:
        assert len(result_holdout) == 1
        result_holdout = result_holdout[0]
    else:
        result_holdout = None
    return result_holdout


def results_to_holdout(result_lst: list[BaselineResult]) -> list[BaselineResult]:
    return [result_to_holdout(result) for result in result_lst]
