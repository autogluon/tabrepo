from __future__ import annotations

import pandas as pd

from tabrepo.scripts_v6.logging_config import setup_logger
from tabrepo import load_repository, EvaluationRepository
from tabrepo.scripts_v6.TabPFN_class import CustomTabPFN
from tabrepo.scripts_v6.TabPFNv2_class import CustomTabPFNv2
from tabrepo.scripts_v6.LGBM_class import CustomLGBM
from experiment_utils import run_experiments, convert_leaderboard_to_configs

logger = setup_logger(log_file_name='temp_script_v6')

if __name__ == '__main__':

    logger.info("Starting execution script...")

    context_name = "D244_F3_C1530_30"
    logger.info(f"Loading repository for context: {context_name}")
    try:
        repo: EvaluationRepository = load_repository(context_name, cache=True)
        logger.info("Repository loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load repository: {e}", exc_info=True)
        raise

    expname = "./initial_experiment_tabpfn_v6"  # folder location of all experiment artifacts
    ignore_cache = True  # set to True to overwrite existing caches and re-run experiments from scratch

    # To run everything:
    # datasets = repo.datasets
    # folds = repo.folds
    folds = [0]
    datasets = [
        "blood-transfusion-service-center",  # binary
        "Australian",  # binary
        "balance-scale",  # multiclass
        # "MIP-2016-regression",  # regression
    ]
    logger.info(f"Selected Datasets: {datasets}")
    logger.info(f"Folds to run: {folds}")

    try:
        tids = [repo.dataset_to_tid(dataset) for dataset in datasets]
    except Exception as e:
        logger.warning(f"Some datasets may not belong to the repository: {e}", exc_info=True)

    methods_dict = {
        "LightGBM": {
            "learning_rate": 0.15,
            "num_leaves": 32,
            "verbose": -1,  # To suppress warnings
        },
        "TabPFN": {
            "device": 'cpu',
            "N_ensemble_configurations": 32,
        },
    }
    method_cls_dict = {
        "LightGBM": CustomLGBM,
        "TabPFN": CustomTabPFN,
        "TabPFNv2": CustomTabPFNv2,
    }
    methods = list(methods_dict.keys())
    logger.info(f"Methods to run: {methods}")

    logger.info("Running experiments...")
    try:
        results_lst = run_experiments(
            expname=expname,
            tids=tids,
            folds=folds,
            methods=methods,
            methods_dict=methods_dict,
            method_cls=method_cls_dict,
            task_metadata=repo.task_metadata,
            ignore_cache=ignore_cache,
        )
        logger.info("Experiments Status: Successful.")
    except Exception as e:
        logger.error(f"An error occurred while running experiments: {e}", exc_info=True)
        raise

    logger.info("Concatenating results into Dataframe...")
    try:
        results_df = pd.concat(results_lst, ignore_index=True)
    except Exception as e:
        logger.error(f"An error occurred while concatenating results: {e}", exc_info=True)

    logger.info("Renaming leaderboard columns... ")
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
    logger.info(f"Comparison configs: {comparison_configs}")

    baselines = [
        "AutoGluon_bq_4h8c_2023_11_14",
    ]
    logger.info(f"Baseline: {baselines}")

    logger.info(f"Comparing metrics...")
    try:
        metrics = repo.compare_metrics(
            results_df,
            datasets=datasets,
            folds=repo.folds,
            baselines=baselines,
            configs=comparison_configs,
        )
    except Exception as e:
        logger.error(f"An error occurred in compare_metrics(): {e}", exc_info=True)
        raise

    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(f"Config Metrics Example:\n{metrics}")

    logger.info("Plotting overall rank comparison...")
    try:
        evaluator_output = repo.plot_overall_rank_comparison(
            results_df=metrics,
            save_dir=expname,
        )
    except Exception as e:
        logger.error(f"An error occurred in plot_overall_rank_comparison(): {e}", exc_info=True)
