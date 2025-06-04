from __future__ import annotations

import pickle
import requests

import pandas as pd

from autogluon.common.loaders import load_pd
from autogluon.common.savers import save_pd, save_pkl

BANNED_DATASETS = [
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


def _load_pickle_from_url(url):
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad status codes
    return pickle.loads(response.content)


def _load_repo(context_name: str):
    from tabrepo import EvaluationRepository, EvaluationRepositoryCollection
    cache_path = f"./{context_name}/repo_cache/tabarena_all.pkl"
    repo = EvaluationRepositoryCollection.load(path=cache_path)
    return repo


def load_results(lite: bool = False) -> pd.DataFrame:
    """
    Simple function to load the results of the TabArena 2025 paper.
    The results are at the per-task level (prior to dataset aggregation).
    For simplicity, the `normalized-error` columns have been removed.

    Parameters
    ----------
    lite: bool, default False
        If True, returns only the first split (fold 0, repeat 0) of the results.

    Returns
    -------
    df_results: pd.DataFrame
        The results on every method benchmarked in the TabArena paper (as present in figure 1)
        Also includes AutoGluon and the simulated Portfolio.

    """
    if lite:
        path = "https://tabarena.s3.us-west-2.amazonaws.com/results/df_results_lite_leaderboard.parquet"
    else:
        path = "https://tabarena.s3.us-west-2.amazonaws.com/results/df_results_leaderboard.parquet"
    df_results = pd.read_parquet(path=path)
    return df_results


def load_paper_results(
    context_name: str = "tabarena_paper_full_51",
    load_from_s3: bool = True,
    generate_from_repo: bool = False,
    save_local_to_s3: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    _s3_prefix = "evaluation"
    s3_prefix_public = f"https://tabarena.s3.us-west-2.amazonaws.com/{_s3_prefix}"
    s3_prefix_private = f"s3://tabarena/{_s3_prefix}"

    path_datasets_tabpfn = f"{context_name}/datasets_tabpfn.pkl"
    path_datasets_tabicl = f"{context_name}/datasets_tabicl.pkl"

    df_result_save_path = f"{context_name}/data/df_results"
    df_result_save_path_holdout = f"{context_name}/data/df_results_holdout"

    df_result_save_path_w_norm_err = f"{df_result_save_path}_w_norm_err"
    df_result_save_path_holdout_w_norm_err = f"{df_result_save_path_holdout}_w_norm_err"

    df_result_save_path += ".parquet"
    df_result_save_path_holdout += ".parquet"
    df_result_save_path_w_norm_err += ".parquet"
    df_result_save_path_holdout_w_norm_err += ".parquet"

    if load_from_s3:
        print(f"Downloading files from s3 (load_from_s3=True). Set to False in future runs for faster runtimes.")
        # Do this for your first run
        df_results = load_pd.load(path=f"{s3_prefix_public}/{df_result_save_path}")
        df_results_holdout = load_pd.load(path=f"{s3_prefix_public}/{df_result_save_path_holdout}")
        save_pd.save(path=df_result_save_path, df=df_results)
        save_pd.save(path=df_result_save_path_holdout, df=df_results_holdout)

        # df_results_w_norm_err = PaperRunTabArena.compute_normalized_error_dynamic(df_results)
        # df_results_holdout_w_norm_err = PaperRunTabArena.compute_normalized_error_dynamic(df_results_holdout)
        df_results_w_norm_err = load_pd.load(path=f"{s3_prefix_public}/{df_result_save_path_w_norm_err}")
        df_results_holdout_w_norm_err = load_pd.load(path=f"{s3_prefix_public}/{df_result_save_path_holdout_w_norm_err}")
        save_pd.save(path=df_result_save_path_w_norm_err, df=df_results_w_norm_err)
        save_pd.save(path=df_result_save_path_holdout_w_norm_err, df=df_results_holdout_w_norm_err)
    else:
        # Do this to save time in future runs
        df_results_w_norm_err = load_pd.load(path=df_result_save_path_w_norm_err)
        df_results_holdout_w_norm_err = load_pd.load(path=df_result_save_path_holdout_w_norm_err)

    if save_local_to_s3:
        # save_pd.save(df=df_results, path=f"{s3_prefix_private}/{df_result_save_path}")
        # save_pd.save(df=df_results_holdout, path=f"{s3_prefix_private}/{df_result_save_path_holdout}")
        save_pd.save(df=df_results_w_norm_err, path=f"{s3_prefix_private}/{df_result_save_path_w_norm_err}")
        save_pd.save(df=df_results_holdout_w_norm_err, path=f"{s3_prefix_private}/{df_result_save_path_holdout_w_norm_err}")

    if generate_from_repo:
        # Requires large repo artifacts downloaded
        repo = _load_repo(context_name=context_name)

        datasets_tabpfn = repo.datasets(configs=["TabPFNv2_c1_BAG_L1"])
        datasets_tabicl = repo.datasets(configs=["TabICL_c1_BAG_L1"])

        datasets_tabpfn = [d for d in datasets_tabpfn if d not in BANNED_DATASETS]
        datasets_tabicl = [d for d in datasets_tabicl if d not in BANNED_DATASETS]

        save_pkl.save(path=f"{s3_prefix_private}/{path_datasets_tabpfn}", object=datasets_tabpfn)
        save_pkl.save(path=f"{s3_prefix_private}/{path_datasets_tabicl}", object=datasets_tabicl)
    else:
        datasets_tabpfn = _load_pickle_from_url(url=f"{s3_prefix_public}/{path_datasets_tabpfn}")
        datasets_tabicl = _load_pickle_from_url(url=f"{s3_prefix_public}/{path_datasets_tabicl}")

        # download_s3_file(s3_bucket="tabarena", s3_prefix=f"{_s3_prefix}/{path_datasets_tabpfn}")
        # datasets_tabpfn = load_pkl.load(path=f"{s3_prefix_public}/{path_datasets_tabpfn}")
        # datasets_tabicl = load_pkl.load(path=f"{s3_prefix_public}/{path_datasets_tabicl}")

    return df_results_w_norm_err, df_results_holdout_w_norm_err, datasets_tabpfn, datasets_tabicl
