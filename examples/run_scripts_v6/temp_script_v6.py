from __future__ import annotations

import logging
import os
import pandas as pd
import datetime as dt

from tabrepo.scripts_v6 import logging_config
from tabrepo.scripts_v6.logging_config import utils_logger as log
from tabrepo import load_repository, EvaluationRepository
from tabrepo.scripts_v6.LGBM_class import CustomLGBM
from tabrepo.utils.experiment_utils_v6 import run_experiments, convert_leaderboard_to_configs


def datetime_iso(datetime=None, date=True, time=True, micros=False, date_sep='-', datetime_sep='T', time_sep=':',
                 micros_sep='.', no_sep=False):
    """

    :param date:
    :param time:
    :param micros:
    :param date_sep:
    :param time_sep:
    :param datetime_sep:
    :param micros_sep:
    :param no_sep: if True then all separators are taken as empty string
    :return:
    """
    if no_sep:
        date_sep = time_sep = datetime_sep = micros_sep = ''
    strf = ""
    if date:
        strf += "%Y{_}%m{_}%d".format(_=date_sep)
        if time:
            strf += datetime_sep
    if time:
        strf += "%H{_}%M{_}%S".format(_=time_sep)
        if micros:
            strf += "{_}%f".format(_=micros_sep)
    datetime = dt.datetime.utcnow() if datetime is None else datetime
    return datetime.strftime(strf)


def output_dirs(root=None, subdirs=None, create=False):
    root = root if root is not None else '.'
    if create:
        os.makedirs(root, exist_ok=True)

    dirs = {
        'root': root,
    }

    if subdirs is not None:
        if isinstance(subdirs, str):
            subdirs = [subdirs]

        for d in subdirs:
            subdir_path = os.path.join(root, d)
            dirs[d] = subdir_path
            if create:
                os.makedirs(subdir_path, exist_ok=True)

    return dirs


script_name = os.path.splitext(os.path.basename(__file__))[0]
now_str = datetime_iso(date_sep='', time_sep='')
log_dir = output_dirs(subdirs='logs', create=True)['logs']
logging_config.setup(log_file=os.path.join(log_dir, '{script}.{now}.log'.format(script=script_name, now=now_str)),
                     root_file=os.path.join(log_dir, '{script}.{now}.full.log'.format(script=script_name, now=now_str)),
                     root_level=logging.DEBUG, app_level=logging.INFO, console_level=logging.INFO, print_to_log=True)

if __name__ == '__main__':

    log.info("Starting execution script...")
    log.debug(f"Logs are stored in: {log_dir}")

    context_name = "D244_F3_C1530_30"
    log.info(f"Loading repository for context: {context_name}")
    try:
        repo: EvaluationRepository = load_repository(context_name, cache=True)
        log.info("Repository loaded successfully.")
    except Exception as e:
        log.error(f"Failed to load repository: {e}", exc_info=True)
        raise

    expname = "./initial_experiment_tabpfn_v6"  # folder location of all experiment artifacts
    ignore_cache = True  # set to True to overwrite existing caches and re-run experiments from scratch

    # To run everything:
    # datasets = repo.datasets
    # folds = repo.folds
    folds = [0]
    # datasets = [
    #     "blood-transfusion-service-center",  # binary
    #     "Australian",  # binary
    #     "balance-scale",  # multiclass
    #     # "MIP-2016-regression",  # regression
    # ]

    datasets = [
        "blood-transfusion-service-center",  # binary
    ]
    log.info(f"Selected Datasets: {datasets}")
    log.info(f"Folds to run: {folds}")

    try:
        tids = [repo.dataset_to_tid(dataset) for dataset in datasets]
    except Exception as e:
        log.warning(f"Some datasets may not belong to the repository: {e}", exc_info=True)

    methods_dict = {
        "LightGBM": {
            "learning_rate": 0.15,
            "num_leaves": 32,
            "verbose": -1,  # To suppress warnings
        },
    }
    method_cls_dict = {
        "LightGBM": CustomLGBM,
    }
    methods = list(methods_dict.keys())
    log.info(f"Methods to run: {methods}")

    log.info("Running experiments...")
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
        log.info("Experiments Status: Successful.")
    except Exception as e:
        log.error(f"An error occurred while running experiments: {e}", exc_info=True)
        raise

    log.info("Concatenating results into Dataframe...")
    try:
        results_df = pd.concat(results_lst, ignore_index=True)
    except Exception as e:
        log.error(f"An error occurred while concatenating results: {e}", exc_info=True)

    log.info("Renaming leaderboard columns... ")
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
    log.info(f"Comparison configs: {comparison_configs}")

    baselines = [
        "AutoGluon_bq_4h8c_2023_11_14",
    ]
    log.info(f"Baseline: {baselines}")

    log.info(f"Comparing metrics...")
    from tabrepo.evaluation.evaluator import Evaluator
    evaluator = Evaluator(repo=repo)
    try:
        metrics = evaluator.compare_metrics(
            results_df,
            datasets=datasets,
            folds=folds,
            baselines=baselines,
            configs=comparison_configs,
        )
    except Exception as e:
        log.error(f"An error occurred in compare_metrics(): {e}", exc_info=True)
        raise

    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(f"Config Metrics Example:\n{metrics}")

    log.info("Plotting overall rank comparison...")
    try:
        evaluator_output = evaluator.plot_overall_rank_comparison(
            results_df=metrics,
            save_dir=expname,
        )
    except Exception as e:
        log.error(f"An error occurred in plot_overall_rank_comparison(): {e}", exc_info=True)
