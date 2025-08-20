from __future__ import annotations

from pathlib import Path

import pandas as pd

from tabrepo import EvaluationRepositoryCollection
from tabrepo.benchmark.result import BaselineResult
from tabrepo.nips2025_utils.artifacts.method_metadata import MethodMetadata
from tabrepo.nips2025_utils.artifacts.tabarena51_artifact_loader import TabArena51ArtifactLoader
from tabrepo.nips2025_utils.tabarena_context import TabArenaContext

"""
This is an example script showcasing how to access the different types of artifacts in TabArena.
Running this script without edits will download all artifacts in TabArena, requiring ~1 TB of disk space.
"""
if __name__ == '__main__':
    loader = TabArena51ArtifactLoader()
    tabarena_context = TabArenaContext()
    methods: list[str] = list(tabarena_context.methods)  # the list of valid methods
    cur_method = "TabDPT_GPU"  # TabDPT is used as an example.
    method_metadata: MethodMetadata = tabarena_context.method_metadata(method=cur_method)

    # Get the hyperparameters for all configs
    configs_hyperparameters: dict[str, dict] = tabarena_context.load_configs_hyperparameters(download="auto")

    # download raw data for all methods, very large (1 TB)
    # raw contains all available information. Uniquely, it contains:
    # 1. test prediction (probabilities) for all inner-fold models and the overall bagged ensemble
    # 2. val prediction (probabilities) for all inner-fold models.
    # 3. test and val scores
    # 4. Model hyperparameters
    # 5. Train time, inference time, total time
    # 6. Available memory, disk space usage, cpu count, gpu count
    # 6. Numerous task metadata information
    # 7. Numerous model metadata information
    # saved to: ~/.cache/tabarena/artifacts/
    loader.download_raw()  # raw for all methods
    # loader._download_raw_method(cur_method)  # raw for just a single method

    # download processed data, much smaller (100 GB)
    # processed data contains the information needed to simulate model portfolios and hyperparameter optimization.
    # processed data is stored in a `EvaluationRepository` object with many quality of life features.
    # We recommend most users to interact with the processed data instead of the raw data.
    # saved to: ~/.cache/tabarena/artifacts/
    loader.download_processed()  # processed for all methods
    # loader._download_processed_method(cur_method)  # processed for just a single method

    # download results data (<100 MB)
    # The results data are stored as pandas DataFrames with (method, dataset, fold) as the unique key.
    # contains the test error, val error, training time, inference time, and more.
    # saved to: ~/.cache/tabarena/artifacts/
    loader.download_results()  # results for all methods
    # loader._download_results_method(cur_method)  # results for just a single method

    # The full raw results for a given method type. Can be very large.
    # Each element in the list corresponds to a specific (method, dataset, split) run.
    # It is recommended to use a debugger to best understand the contents of the raw artifact.
    results_raw_lst: list[BaselineResult] = method_metadata.load_raw()

    df_results: pd.DataFrame = tabarena_context.load_results_paper(methods=methods, download_results="auto")

    for method in methods:
        # convert raw to processed, unnecessary if already called `loader.download_processed()`
        path_to_repo_artifact: Path = tabarena_context.generate_repo(method=method)

    # load all processed data
    repo: EvaluationRepositoryCollection = tabarena_context.load_repo(methods=methods)

    for method in methods:
        # convert processed to results, unnecessary if already called `loader.download_results()`
        results, config_results = tabarena_context.simulate_repo(method=method)
