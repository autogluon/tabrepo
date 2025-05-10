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
        "repos/tabarena60_autogluon",
        "repos/tabarena60_tabicl",
        "repos/tabarena60_tabpfnv2",
    ]
    repos_extra = [
        "Dummy",
        "FTTransformer",
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
    ]

    repos_extra = [f"repos/tabarena61/{r}" for r in repos_extra]

    repo_names = repo_names + repos_extra

    repos = [EvaluationRepository.from_dir(repo_path) for repo_path in repo_names]
    repo = EvaluationRepositoryCollection(repos=repos)

    if toy_mode:
        repo = repo.subset(folds=[0])
    return repo


if __name__ == '__main__':
    toy_mode = False
    load_cache = True
    load_sim_cache = True

    context_name = "tabarena_all"
    if toy_mode:
        context_name += "_toy"
    cache_path = f"./repo_cache/{context_name}"
    df_result_save_path = f"./tmp/{context_name}/df_results.parquet"
    eval_save_path = "tabarena_paper"
    if toy_mode:
        cache_path += "_toy"
    cache_path += ".pkl"

    if load_cache:
        repo = EvaluationRepositoryCollection.load(path=cache_path)
    else:
        repo = load_repo(toy_mode=toy_mode)
        repo.save(path=cache_path)
    # repo = repo.subset(folds=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    repo.set_config_fallback(config_fallback="RandomForest_c1_BAG_L1")

    paper_run = PaperRunTabArena(repo=repo, output_dir=eval_save_path)
    if not load_sim_cache:
        df_results = paper_run.run()
        save_pd.save(df=df_results, path=df_result_save_path)
    else:
        df_results = load_pd.load(path=df_result_save_path)

    df_results = paper_run.compute_normalized_error(df_results=df_results)
    paper_run.eval(df_results=df_results)

    # sanity_check(repo=repo, fillna=True, filter_to_all_valid=False, results_df=results_df, results_df_extra=results_hpo)
