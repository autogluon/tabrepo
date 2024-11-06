from __future__ import annotations

import pandas as pd

from experiment_utils import ExperimentBatchRunner
from experiment_runner import OOFExperimentRunner
from tabrepo import EvaluationRepository, EvaluationRepositoryCollection
from tabrepo.scripts_v5.AutoGluon_class import AGWrapper
from tabrepo.utils.cache import SimulationExperiment


if __name__ == '__main__':
    # Load Context
    context_name = "D244_F3_C1530_30"  # 30 smallest datasets. To run larger, set to "D244_F3_C1530_200"
    expname = "./initial_experiment_simple_simulator"  # folder location of all experiment artifacts
    ignore_cache = False  # set to True to overwrite existing caches and re-run experiments from scratch

    repo_og: EvaluationRepository = EvaluationRepository.from_context(context_name, cache=True)

    task_metadata = repo_og.task_metadata.copy(deep=True)

    # Sample for a quick demo
    datasets = repo_og.datasets()[:3]
    folds = [0, 1, 2]

    # To run everything:
    # datasets = repo.datasets()
    # folds = repo.folds

    methods = [
        ("LightGBM_c1_BAG_L1_Reproduced", AGWrapper, {"fit_kwargs": {
            "num_bag_folds": 8, "num_bag_sets": 1, "fit_weighted_ensemble": False, "calibrate": False, "verbosity": 0,
            "hyperparameters": {"GBM": [{}]},
        }}),
    ]

    tids = [repo_og.dataset_to_tid(dataset) for dataset in datasets]
    repo: EvaluationRepository = ExperimentBatchRunner().generate_repo_from_experiments(
        expname=expname,
        tids=tids,
        folds=folds,
        methods=methods,
        experiment_cls=OOFExperimentRunner,
        cache_cls=SimulationExperiment,
        task_metadata=repo_og.task_metadata,
        ignore_cache=ignore_cache,
    )

    repo.print_info()

    save_path = "repo_new"
    repo.to_dir(path=save_path)  # Load the repo later via `EvaluationRepository.from_dir(save_path)`

    print(f"New Configs   : {repo.configs()}")

    repo_combined = EvaluationRepositoryCollection(repos=[repo_og, repo], config_fallback="ExtraTrees_c1_BAG_L1")
    repo_combined = repo_combined.subset(datasets=repo.datasets(), folds=repo.folds)

    repo_combined.print_info()

    comparison_configs = [
        "RandomForest_c1_BAG_L1",
        "ExtraTrees_c1_BAG_L1",
        "LightGBM_c1_BAG_L1",
        "XGBoost_c1_BAG_L1",
        "CatBoost_c1_BAG_L1",
        "NeuralNetTorch_c1_BAG_L1",
        "NeuralNetFastAI_c1_BAG_L1",
        "LightGBM_c1_BAG_L1_Reproduced",
    ]

    baselines = [
        "AutoGluon_bq_4h8c_2023_11_14",
    ]

    metrics = repo_combined.compare_metrics(
        datasets=datasets,
        folds=folds,
        baselines=baselines,
        configs=comparison_configs,
    )

    metrics_tmp = metrics.reset_index(drop=False)

    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(f"Config Metrics Example:\n{metrics.head(100)}")

    evaluator_output = repo_combined.plot_overall_rank_comparison(
        results_df=metrics,
        save_dir=expname,
    )
