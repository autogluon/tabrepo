from __future__ import annotations

import pickle
import requests

import pandas as pd

from autogluon.common.loaders import load_pd, load_pkl
from autogluon.common.savers import save_pd, save_pkl
from autogluon.common.utils.s3_utils import download_s3_file


def _load_pickle_from_url(url):
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad status codes
    return pickle.loads(response.content)


def _load_repo(context_name: str):
    from tabrepo import EvaluationRepository, EvaluationRepositoryCollection
    cache_path = f"./{context_name}/repo_cache/tabarena_all.pkl"
    repo = EvaluationRepositoryCollection.load(path=cache_path)
    return repo


def load_paper_results(
    context_name: str = "tabarena_paper_full_gpu",
    load_from_s3: bool = True,
    generate_from_repo: bool = False,
    norm_error_static: bool = False,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    _s3_prefix = "evaluation"
    s3_prefix_public = f"https://tabarena.s3.us-west-2.amazonaws.com/{_s3_prefix}"
    s3_prefix_private = f"s3://tabarena/{_s3_prefix}"

    path_datasets_tabpfn = f"{context_name}/datasets_tabpfn.pkl"
    path_datasets_tabicl = f"{context_name}/datasets_tabicl.pkl"

    df_result_save_path_w_norm_err = f"{context_name}/data/df_results_w_norm_err"
    eval_save_path = f"{context_name}/output"

    if norm_error_static:
        eval_save_path += "_static"
        df_result_save_path_w_norm_err += "_static"
    df_result_save_path_w_norm_err += ".parquet"

    if load_from_s3:
        # Do this for your first run
        df_results = load_pd.load(path=f"{s3_prefix_public}/{df_result_save_path_w_norm_err}")
        save_pd.save(path=df_result_save_path_w_norm_err, df=df_results)
    else:
        # Do this to save time in future runs
        df_results = load_pd.load(path=df_result_save_path_w_norm_err)

    # paper_run = PaperRunTabArena(repo=None, output_dir=eval_save_path)
    # df_results = paper_run.compute_normalized_error(df_results=df_results)

    if generate_from_repo:
        # Requires large repo artifacts downloaded
        repo = _load_repo(context_name=context_name)
        datasets_tabpfn = repo.datasets(configs=["TabPFNv2_c1_BAG_L1"])
        datasets_tabicl = repo.datasets(configs=["TabICL_c1_BAG_L1"])
        save_pkl.save(path=f"{s3_prefix_private}/{path_datasets_tabpfn}", object=datasets_tabpfn)
        save_pkl.save(path=f"{s3_prefix_private}/{path_datasets_tabicl}", object=datasets_tabicl)
    else:
        datasets_tabpfn = _load_pickle_from_url(url=f"{s3_prefix_public}/{path_datasets_tabpfn}")
        datasets_tabicl = _load_pickle_from_url(url=f"{s3_prefix_public}/{path_datasets_tabicl}")

        # download_s3_file(s3_bucket="tabarena", s3_prefix=f"{_s3_prefix}/{path_datasets_tabpfn}")
        # datasets_tabpfn = load_pkl.load(path=f"{s3_prefix_public}/{path_datasets_tabpfn}")
        # datasets_tabicl = load_pkl.load(path=f"{s3_prefix_public}/{path_datasets_tabicl}")

    return df_results, datasets_tabpfn, datasets_tabicl
