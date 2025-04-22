from __future__ import annotations

import pandas as pd

from tabrepo import EvaluationRepository, Evaluator
from tabrepo.utils.pickle_utils import load_all_pickles
from tabrepo.benchmark.result import ExperimentResults


"""
# INSTRUCTIONS: First run the below commands in the folder 1 above the tabrepo root folder to fetch the data from S3 (ex: `workspace/code`)
# NOTE: These files are non-public, and cant be accessed without credentials

S3_BUCKET=prateek-ag
EXP_NAME=neerick-exp-big-realmlp-alt
EXCLUDE=(--exclude "*.log" --exclude "*.json")

EXP_DATA_PATH=${EXP_NAME}/data/
S3_DIR=s3://${S3_BUCKET}/${EXP_DATA_PATH}
USER_DIR=../data/${EXP_DATA_PATH}
echo "${S3_DIR} ${USER_DIR} ${EXCLUDE[@]}"
aws s3 cp --recursive ${S3_DIR} ${USER_DIR} ${EXCLUDE[@]}
"""


if __name__ == '__main__':
    # Load Context
    expname = "../../../data/neerick-exp-big-realmlp-alt"  # folder location of results, need to point this to the correct folder
    repo_dir = "repos/tabarena_big_s3_realmlp_alt"  # location of local cache for fast script running
    load_repo = False  # ensure this is False for the first time running

    # The original TabRepo artifacts for the 1530 configs
    context_name = "D244_F3_C1530_200"
    repo_og: EvaluationRepository = EvaluationRepository.from_context(context_name, load_predictions=False)

    if not load_repo:
        exp_results = ExperimentResults(task_metadata=repo_og.task_metadata)

        # load all pickles into a directory
        results_lst: list[dict] = load_all_pickles(dir_path=expname)

        # Convert the run artifacts into an EvaluationRepository
        repo: EvaluationRepository = exp_results.repo_from_results(results_lst=results_lst)
        repo.to_dir(repo_dir)
    else:
        repo = EvaluationRepository.from_dir(repo_dir)
    repo.print_info()

    new_baselines = repo.baselines()
    new_configs = repo.configs()
    print(f"New Baselines : {new_baselines}")
    print(f"New Configs   : {new_configs}")
    print(f"New Configs Hyperparameters: {repo.configs_hyperparameters()}")

    # create an evaluator to compute comparison metrics such as win-rate and ELO
    evaluator = Evaluator(repo=repo)
    metrics = evaluator.compare_metrics(
        baselines=new_baselines,
        configs=new_configs,
        fillna=False,
    )
    metrics = metrics.reset_index(drop=False)

    from tabrepo.tabarena.tabarena import TabArena
    tabarena = TabArena(
        method_col="framework",
        task_col="dataset",
        seed_column="fold",
        error_col="metric_error",
        columns_to_agg_extra=[
            "time_train_s",
            "time_infer_s",
        ],
        groupby_columns=[
            "metric",
            "problem_type",
        ]
    )

    metrics_fillna = tabarena.fillna_data(data=metrics)

    leaderboard = tabarena.leaderboard(
        data=metrics_fillna,
    )

    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(leaderboard)
