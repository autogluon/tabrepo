import ast
import copy
from typing import List

import numpy as np
from dataclasses import dataclass

import pandas as pd
import ray

from syne_tune.experiments import load_experiments_df
from tqdm import tqdm

from autogluon_zeroshot.portfolio.zeroshot_selection import zeroshot_configs
from autogluon_zeroshot.repository import EvaluationRepository
from autogluon_zeroshot.utils import loop_utils


@dataclass
class ResultRow:
    taskid: int  # OpenML taskid, also refered to "tid"
    fold: int
    method: str
    test_error: float
    rank: float
    normalized_score: float
    config_selected: list = None


def automl_results(repo: EvaluationRepository, dataset_names: List[str], n_folds: int, rank_scorer, normalized_scorer) -> List[ResultRow]:
    """
    :return: evaluation of AutoGluon medium/high/best quality.
    """
    automl_df = copy.deepcopy(repo._zeroshot_context.df_results_by_dataset_automl)
    automl_df['fold'] = automl_df['dataset'].map(repo._zeroshot_context.dataset_name_to_fold_dict)
    automl_df['tid'] = automl_df['dataset'].map(repo._zeroshot_context.dataset_name_to_tid_dict)
    automl_df['task'] = automl_df['dataset']
    # FIXME: Instead of returning list of "ResultRow", return this dataframe
    automl_df['dataset'] = automl_df['tid'].map(repo._zeroshot_context.tid_to_dataset_dict)

    rows_automl = []
    for dataset in tqdm(dataset_names):
        for fold in range(n_folds):
            dataset_fold_name = f"{repo.dataset_to_taskid(dataset)}_{fold}"
            automl_df_fold = automl_df[automl_df['task'] == dataset_fold_name]
            task_automl_dict = automl_df_fold.T.to_dict()

            for k, v in task_automl_dict.items():
                metric_error = v['metric_error']
                rows_automl.append(ResultRow(
                    taskid=v['tid'],
                    fold=v['fold'],
                    method=v['framework'],
                    test_error=metric_error,
                    rank=rank_scorer.rank(dataset_fold_name, metric_error),
                    normalized_score=normalized_scorer.rank(dataset_fold_name, metric_error),
                ))

    return rows_automl


def _zeroshot_results(i,
                      repo: EvaluationRepository,
                      df_rank: pd.DataFrame,
                      train_dataset_names: List[str],  # FIXME: Use this instead of assuming LOO
                      test_dataset_names: List[str],  # FIXME: Allow more than 1 element
                      n_folds: int,  # TODO: Switch to list of folds to have more control
                      rank_scorer,
                      normalized_scorer,
                      ensemble_size: int,
                      portfolio_size,
                      name_suffix: str) -> List[ResultRow]:
    rows_zeroshot = []
    if name_suffix is None:
        name_suffix = ''

    assert len(test_dataset_names) == 1
    dataset = test_dataset_names[0]

    taskid = repo.dataset_to_taskid(dataset)
    train_datasets = [x for x in df_rank.columns if x.split("_")[0] != str(taskid)]
    # assert train_datasets == train_dataset_names

    indices = zeroshot_configs(-df_rank[train_datasets].values.T, portfolio_size)
    portfolio_configs = [df_rank.index[i] for i in indices]

    # run best base model and ensemble
    suffix = f"-{portfolio_size}-{ensemble_size}"
    if ensemble_size:
        suffix += " (ensemble)"
    test_errors = repo.evaluate_ensemble(
        dataset_names=[dataset],
        config_names=portfolio_configs,
        ensemble_size=ensemble_size,
        rank=False,
        backend='native',
    )
    assert test_errors.shape[0] == 1  # we send one model, we should get one row back
    for fold in range(n_folds):
        test_error = test_errors[0][fold]
        dataset_fold_name = f"{repo.dataset_to_taskid(dataset)}_{fold}"
        rows_zeroshot.append(ResultRow(
            taskid=taskid,
            fold=fold,
            method=f"Zeroshot{name_suffix}{suffix}",
            test_error=test_error,
            rank=rank_scorer.rank(dataset_fold_name, test_error),
            normalized_score=normalized_scorer.rank(dataset_fold_name, test_error),
            config_selected=portfolio_configs,
        ))
    print(f'Done {i}: {dataset}')
    return rows_zeroshot


def zeroshot_results(
    repo: EvaluationRepository,
    dataset_names: List[str],
    n_folds: int,
    rank_scorer,
    normalized_scorer,
    configs: List[str] = None,
    ensemble_sizes: List[int] = [1, 20],
    portfolio_sizes: List[int] = [5, 10, 20, 40, 80],
    name_suffix: str = None,
    backend: str = 'ray',
):
    """
    :param ensemble_sizes: number of caruana sizes to consider
    :param portfolio_sizes: number of portfolio to consider
    :return: evaluation obtained on all combinations from `ensemble_sizes` and `portfolio_sizes`
    """
    dd = repo._zeroshot_context.df_results_by_dataset_vs_automl
    df_rank = dd.pivot_table(index="framework", columns="dataset", values="score_val").rank()
    df_rank.fillna(value=np.nanmax(df_rank.values), inplace=True)
    assert not any(df_rank.isna().values.reshape(-1))

    if configs is not None:
        df_rank = df_rank.loc[configs]
        assert len(df_rank) == len(configs)

    if backend == 'ray':
        if not ray.is_initialized():
            ray.init()
        repo = ray.put(repo)
        df_rank = ray.put(df_rank)
        rank_scorer = ray.put(rank_scorer)
        normalized_scorer = ray.put(normalized_scorer)
        loop_func = loop_utils.with_ray
    elif backend == 'seq':
        loop_func = loop_utils.with_seq
    else:
        raise ValueError(f'Invalid backend: "{backend}"')

    i = 1
    input_list = []
    kwargs = {
        'repo': repo,
        'df_rank': df_rank,
        'train_dataset_names': None,
        'rank_scorer': rank_scorer,
        'normalized_scorer': normalized_scorer,
        'name_suffix': name_suffix,

    }
    for dataset in dataset_names:
        for portfolio_size in portfolio_sizes:
            for ensemble_size in ensemble_sizes:
                # Create and execute all tasks in parallel
                input_row_dict = {
                    'i': i,
                    'test_dataset_names': [dataset],
                    'n_folds': n_folds,
                    'ensemble_size': ensemble_size,
                    'portfolio_size': portfolio_size,
                }
                input_list.append(input_row_dict)
                i += 1
    print(f'total: {i}')
    rows_zeroshot: list = loop_func(
        func=_zeroshot_results,
        input_list=input_list,
        kwargs=kwargs,
    )
    rows_zeroshot = [item for sublist in rows_zeroshot for item in sublist]
    return rows_zeroshot


def evaluate_tuning(
        repo: EvaluationRepository, dataset_names: List[str], n_folds: int, rank_scorer, normalized_scorer,
        expname="02-05-v2",
) -> List[ResultRow]:
    """
    :param expname: name of the experiment tag passed when launching `scripts/method_comparison/run_method_comparison.py`
    :return: results of top configurations found by local search and zeroshot when using LocalSearch, Zeroshot, Zeroshot ensemble.
    """
    def load_df_tuning(expname):
        name_filter = lambda path: expname in str(path)
        df_results = load_experiments_df(path_filter=name_filter)
        df_results["fold"] = df_results.apply(lambda row: int(row['tuner_name'].split("fold-")[1].split("-")[0]),
                                              axis=1)
        for col in ['configs', 'train_datasets', 'test_datasets']:
            df_results[col] = df_results[f'config_{col}'].apply(lambda x: ast.literal_eval(x))
        return df_results

    def extract_configs_from_tuning_results(df_results):
        rows = []
        for fold in sorted(df_results.fold.unique()):
            df_sub = df_results[(df_results.fold == fold) & (df_results.searcher == "localsearch")]
            # zeroshot config is always the first trial
            row_zeroshot = df_sub.loc[
                df_sub.trial_id == 0, ['configs', 'train_datasets', 'test_datasets', 'train_error', 'test_error']].head(
                1)
            # localsearch config is the one with lowest train error
            row_localsearch = df_sub.sort_values("train_error").loc[:,
                              ['configs', 'train_datasets', 'test_datasets', 'train_error', 'test_error']].head(1)
            assert row_localsearch["train_datasets"].values[0] == row_zeroshot["train_datasets"].values[0]
            rows.append({
                "metafold": fold,
                "zeroshot": row_zeroshot["configs"].values[0],
                "localsearch": row_localsearch["configs"].values[0],
                "train_datasets": row_localsearch["train_datasets"].values[0],
                "test_datasets": row_localsearch["test_datasets"].values[0],
            })
        return rows

    def taskid_to_config(tuning_rows, taskid):
        contains_task = lambda tasks: any(task.split("_")[0] == str(taskid) for task in tasks)
        matches = [row for row in tuning_rows if not contains_task(row['train_datasets'])]
        assert len(matches) >= 1
        return matches[0]

    df_results = load_df_tuning(expname=expname)
    if len(df_results) == 0:
        print(f"No tuning result could be found from the experiment tag passed {expname}, please run run_method_comparison.py with this tag first.")
        return

    tuning_rows = extract_configs_from_tuning_results(df_results)

    rows = []
    for dataset in tqdm(dataset_names):
        for suffix, ensemble_size in [("", 1), (f" (ensemble)", 20)]:
            for method in ["zeroshot", "localsearch"]:
                test_errors = repo.evaluate_ensemble(
                    dataset_names=[dataset],
                    config_names=taskid_to_config(tuning_rows, repo.dataset_to_taskid(dataset))[method],
                    ensemble_size=ensemble_size,
                    rank=False,
                )
                assert test_errors.shape[0] == 1  # we send one model, we should get one row back
                for fold in range(n_folds):
                    test_error = test_errors[0][fold]
                    dataset_fold_name = f"{repo.dataset_to_taskid(dataset)}_{fold}"
                    rows.append(ResultRow(
                        taskid=repo.dataset_to_taskid(dataset),
                        fold=fold,
                        method=f"{method}{suffix}".capitalize(),
                        test_error=test_error,
                        rank=rank_scorer.rank(dataset_fold_name, test_error),
                        normalized_score=normalized_scorer.rank(dataset_fold_name, test_error),
                    ))
    return rows
