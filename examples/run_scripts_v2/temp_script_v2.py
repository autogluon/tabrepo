from __future__ import annotations

import pandas as pd

from exec_v2 import fit_outer
from temp_script_ag_models import convert_leaderboard_to_configs

from autogluon_benchmark.tasks.experiment_utils import run_experiments
from tabrepo import load_repository, EvaluationRepository


if __name__ == '__main__':
    # Load Context
    context_name = "D244_F3_C1530_30"
    repo: EvaluationRepository = load_repository(context_name, cache=True)

    expname = "./initial_experiment_tabpfn"  # folder location of all experiment artifacts
    ignore_cache = False  # set to True to overwrite existing caches and re-run experiments from scratch

    datasets = [
        "blood-transfusion-service-center",
        "Australian",
    ]
    tids = [repo.dataset_to_tid(dataset) for dataset in datasets]
    folds = repo.folds

    methods_dict_tabpfn = {"TabPFN": {}}
    methods_tabpfn = list(methods_dict_tabpfn.keys())

    results_lst = run_experiments(
        expname=expname,
        tids=tids,
        folds=folds,
        methods=methods_tabpfn,
        methods_dict=methods_dict_tabpfn,
        exec_func=fit_outer,
        task_metadata=repo.task_metadata,
        ignore_cache=ignore_cache,
    )

    results_df = pd.concat(results_lst, ignore_index=True)
    results_df = convert_leaderboard_to_configs(results_df)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(results_df)

    comparison_configs = [
        "RandomForest_c1_BAG_L1",
        "ExtraTrees_c1_BAG_L1",
        "LightGBM_c1_BAG_L1",
        "XGBoost_c1_BAG_L1",
        "CatBoost_c1_BAG_L1",
        "TabPFN_c1_BAG_L1",
        "NeuralNetTorch_c1_BAG_L1",
        "NeuralNetFastAI_c1_BAG_L1",
    ]

    metrics = repo.compare_metrics(
        results_df,
        datasets=datasets,
        folds=folds,
        baselines=["AutoGluon_bq_4h8c_2023_11_14"],
        configs=comparison_configs,
    )
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(f"Config Metrics Example:\n{metrics}")
    evaluator_output = repo.plot_overall_rank_comparison(
        results_df=metrics,
        save_dir=expname,
    )
