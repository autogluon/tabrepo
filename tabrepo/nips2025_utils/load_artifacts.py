from __future__ import annotations

from pathlib import Path

from autogluon.common.loaders import load_pkl
from tabrepo.benchmark.result import AGBagResult, BaselineResult
from tabrepo.utils.parallel_for import parallel_for


def load_and_align(path, convert_to_holdout: bool = False) -> BaselineResult:
    data: dict | BaselineResult = load_pkl.load(path)
    data_aligned = BaselineResult.from_dict(data)
    if convert_to_holdout:
        assert isinstance(data_aligned, AGBagResult)
        result_holdout = data_aligned.bag_artifacts(as_baseline=False)
        if len(result_holdout) > 0:
            assert len(result_holdout) == 1
            result_holdout = result_holdout[0]
        else:
            result_holdout = None
        return result_holdout

    # # FIXME: need to figure out why this is not in the artifact by itself
    # #    This does not work, as it is triggered even for non name-suffix cases...
    # if hasattr(data_aligned, "name_suffix"):
    #     from tabrepo.benchmark.preprocessing.preprocessing_register import PREPROCESSING_METHODS
    #
    #     name_suffix = data_aligned.name_suffix
    #     for method_name in PREPROCESSING_METHODS:
    #         if method_name in name_suffix:
    #             name_suffix = "_" + method_name
    #             break
    #     if name_suffix not in data_aligned.framework:
    #         data_aligned.update_name(name_suffix=name_suffix)
    #     if name_suffix not in data_aligned.model_type:
    #         data_aligned.update_model_type(name_suffix=name_suffix)
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
    )
    return results_lst
