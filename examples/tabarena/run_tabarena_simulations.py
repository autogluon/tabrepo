from __future__ import annotations

from tabrepo.paper.paper_runner_tabarena import PaperRunTabArena
from tabrepo import EvaluationRepository, EvaluationRepositoryCollection
from autogluon.common.loaders import load_pd
from autogluon.common.savers import save_pd


"""
# INSTRUCTIONS: First run the below commands in the folder 1 above the tabrepo root folder to fetch the data from S3 (ex: `workspace/code`)
# NOTE: These files are non-public, and cant be accessed without credentials

S3_BUCKET=prateek-ag
EXP_NAME=neerick-exp-tabarena60_big
EXCLUDE=(--exclude "*.log" --exclude "*.json")

EXP_DATA_PATH=${EXP_NAME}/data/
S3_DIR=s3://${S3_BUCKET}/${EXP_DATA_PATH}
USER_DIR=../data/${EXP_DATA_PATH}
echo "${S3_DIR} ${USER_DIR} ${EXCLUDE[@]}"
aws s3 cp --recursive ${S3_DIR} ${USER_DIR} ${EXCLUDE[@]}

S3_BUCKET=prateek-ag
EXP_NAME=neerick-exp-tabarena60_big_v2
EXCLUDE=(--exclude "*.log" --exclude "*.json")

EXP_DATA_PATH=${EXP_NAME}/data/
S3_DIR=s3://${S3_BUCKET}/${EXP_DATA_PATH}
USER_DIR=../data/${EXP_DATA_PATH}
echo "${S3_DIR} ${USER_DIR} ${EXCLUDE[@]}"
aws s3 cp --recursive ${S3_DIR} ${USER_DIR} ${EXCLUDE[@]}

"""


def load_repo(toy_mode: bool):
    repo_names = [
        # "repos/tabarena60_autogluon",
    ]

    repo_names += [
        "repos/tabarena_tabicl_gpu",
        "repos/tabarena_tabpfnv2_gpu",
        "repos/tabarena_tabdpt_gpu",
    ]

    repos_extra = [
        "AutoGluon",

        "Dummy",
        # "FTTransformer",
        "CatBoost",
        "XGBoost",
        "LightGBM",
        "ExplainableBM",
        "ExtraTrees",

        "NeuralNetFastAI",
        "NeuralNetTorch",
        "RandomForest",
        "KNeighbors",
        "LinearModel",
        "RealMLP",

        "TabM",
        "ModernNCA",
    ]

    repos_extra = [f"repos/tabarena61/{r}" for r in repos_extra]

    repo_names = repo_names + repos_extra

    repos = [EvaluationRepository.from_dir(repo_path) for repo_path in repo_names]
    repo = EvaluationRepositoryCollection(repos=repos)

    if toy_mode:
        repo = convert_repo_to_toy(repo=repo, configs_per_type=10)
    return repo


def load_repo_holdout(toy_mode: bool = False):
    repos_extra = [
        "Dummy",
        # "FTTransformer",
        "CatBoost",
        "XGBoost",
        "LightGBM",
        "ExplainableBM",
        "ExtraTrees",

        "NeuralNetFastAI",
        "NeuralNetTorch",
        "RandomForest",
        # "KNeighbors",
        "LinearModel",
        "RealMLP",

        "TabM",
        "ModernNCA",
    ]

    repo_names = [f"repos/tabarena61/holdout/{r}" for r in repos_extra]

    repos = [EvaluationRepository.from_dir(repo_path) for repo_path in repo_names]
    repo = EvaluationRepositoryCollection(repos=repos)

    if toy_mode:
        repo = convert_repo_to_toy(repo=repo, configs_per_type=10)
    return repo


def convert_repo_to_toy(repo: EvaluationRepository, configs_per_type: int = 10, folds: list[int] = None) -> EvaluationRepository:
    if folds is None:
        folds = [0]
    repo = repo.subset(folds=folds)
    configs = repo.configs()
    configs_type_inverse = dict()
    configs_type = repo.configs_type(configs=configs)
    for c in configs:
        c_type = configs_type[c]
        if c_type not in configs_type_inverse:
            configs_type_inverse[c_type] = []
        configs_type_inverse[c_type].append(c)
    configs_toy = []
    for c_type in configs_type_inverse:
        configs_toy += configs_type_inverse[c_type][:configs_per_type]
    repo = repo.subset(configs=configs_toy)
    return repo


def find_missing(repo: EvaluationRepository):
    tasks = repo.tasks()

    n_tasks = len(tasks)

    metrics = repo.metrics()
    metrics = metrics.reset_index(drop=False)

    configs = repo.configs()

    n_configs = len(configs)

    runs_missing_lst = []

    fail_dict = {}
    for i, config in enumerate(configs):
        metrics_config = metrics[metrics["framework"] == config]
        n_tasks_config = len(metrics_config)

        tasks_config = list(metrics_config[["dataset", "fold"]].values)
        tasks_config = set([tuple(t) for t in tasks_config])


        n_tasks_missing = n_tasks - n_tasks_config
        if n_tasks_missing != 0:
            tasks_missing = [t for t in tasks if t not in tasks_config]
        else:
            tasks_missing = []

        for dataset, fold in tasks_missing:
            runs_missing_lst.append(
                (dataset, fold, config)
            )

        print(f"{n_tasks_missing}\t{config}\t{i+1}/{n_configs}")
        fail_dict[config] = n_tasks_missing

    import pandas as pd
    # fail_series = pd.Series(fail_dict).sort_values()

    df_missing = pd.DataFrame(data=runs_missing_lst, columns=["dataset", "fold", "framework"])
    print(df_missing)

    # save_pd.save(path="missing_runs.csv", df=df_missing)

    return df_missing


if __name__ == '__main__':
    toy_mode = False
    load_cache = True
    load_sim_cache = False
    norm_error_static = False
    ban_datasets: bool = True

    context_name = "tabarena_paper_full_gpu"
    if toy_mode:
        context_name += "_toy"
    cache_path = f"./{context_name}/repo_cache/tabarena_all.pkl"
    cache_path_holdout = f"./{context_name}/repo_cache/tabarena_holdout.pkl"

    if ban_datasets:
        df_results_prefix = "51/"
    else:
        df_results_prefix = ""

    df_result_save_path = f"./{context_name}/data/{df_results_prefix}df_results.parquet"
    df_result_save_path_holdout = f"./{context_name}/data/{df_results_prefix}df_results_holdout.parquet"

    norm_err_suffix = ""
    if norm_error_static:
        norm_err_suffix = "_static"

    df_result_save_path_w_norm_err = f"./{context_name}/data/df_results_w_norm_err{norm_err_suffix}.parquet"
    s3_path_df_result_save_path_w_norm_err = f"s3://tabarena/evaluation/{context_name}/data/df_results_w_norm_err{norm_err_suffix}.parquet"
    eval_save_path = f"{context_name}/output{norm_err_suffix}"
    eval_save_path_holdout = f"{context_name}/output_holdout{norm_err_suffix}"

    if load_cache:
        repo = EvaluationRepositoryCollection.load(path=cache_path)
    else:
        repo = load_repo(toy_mode=toy_mode)
        repo.save(path=cache_path)

    # df_missing = find_missing(repo=repo)

    # df_missing = load_pd.load(path="missing_runs.csv")

    banned_datasets = [
        "ASP-POTASSCO",
        "Mobile_Price",
        "Pumpkin_Seeds",
        "abalone",
        "fifa",
        "internet_firewall",
        "cars",
        "steel-plates-fault",
        "solar_flare",
        "PhishingWebsites",
    ]
    repo_datasets = repo.datasets()
    repo_datasets = [d for d in repo_datasets if d not in banned_datasets]
    assert len(repo_datasets) == 51
    if ban_datasets:
        repo = repo.subset(datasets=repo_datasets)

    # df_missing = df_missing[~df_missing["dataset"].isin(banned_datasets)]
    #
    # config_missing_counts = df_missing.value_counts("framework")
    # configs_to_rerun = set(list(config_missing_counts[config_missing_counts < 10].index.values))
    # config_tasks_to_rerun = df_missing[df_missing["framework"].isin(configs_to_rerun)].reset_index(drop=True)
    # save_pd.save(path="s3://tabarena/tmp/rerun_tasks.parquet", df=config_tasks_to_rerun)
    # find_missing(repo=repo)
    # repo = repo.subset(folds=[0, 1, 2, 3, 4, 5, 6, 7, 8])

    # repo = convert_repo_to_toy(repo=repo, configs_per_type=40, folds=[0, 1, 2])

    # raise AssertionError
    repo.set_config_fallback(config_fallback="RandomForest_c1_BAG_L1")

    paper_run = PaperRunTabArena(repo=repo, output_dir=eval_save_path)
    paper_run.generate_data_analysis()

    if not load_sim_cache:
        df_results = paper_run.run()
        save_pd.save(df=df_results, path=df_result_save_path)
        df_results = paper_run.compute_normalized_error(df_results=df_results, static=norm_error_static)
        save_pd.save(df=df_results, path=df_result_save_path_w_norm_err)
    else:
        df_results = load_pd.load(path=df_result_save_path_w_norm_err)

    paper_run = PaperRunTabArena(repo=None, output_dir=eval_save_path)
    paper_run.eval(df_results=df_results)

    # sanity_check(repo=repo, fillna=True, filter_to_all_valid=False, results_df=results_df, results_df_extra=results_hpo)

    # repo_holdout = load_repo_holdout(toy_mode=toy_mode)
    # repo_holdout.save(path=cache_path_holdout)
    repo_holdout = EvaluationRepositoryCollection.load(path=cache_path_holdout)
    if ban_datasets:
        repo_holdout = repo_holdout.subset(datasets=repo_datasets)
    repo_holdout.set_config_fallback(config_fallback="RandomForest_r1_BAG_L1_HOLDOUT")

    paper_run_holdout = PaperRunTabArena(repo=repo_holdout, output_dir=eval_save_path_holdout)
    df_results_holdout = paper_run_holdout.run()
    save_pd.save(df=df_results_holdout, path=df_result_save_path_holdout)
