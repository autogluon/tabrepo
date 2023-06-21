import ast
import copy
import itertools
from typing import List

import numpy as np
from dataclasses import dataclass

from syne_tune.experiments import load_experiments_df
from tqdm import tqdm

from autogluon_zeroshot.portfolio.zeroshot_selection import zeroshot_configs
from autogluon_zeroshot.repository import EvaluationRepository
from autogluon_zeroshot.repository.utils import filter_configs_by_runtime, sort_by_runtime
from autogluon_zeroshot.utils.parallel_for import parallel_for


@dataclass
class ResultRow:
    taskid: int  # OpenML taskid, also refered to "tid"
    fold: int
    method: str
    test_error: float
    rank: float
    normalized_score: float
    config_selected: list = None


def automl_results(repo: EvaluationRepository, dataset_names: List[str], n_eval_folds: int, rank_scorer, normalized_scorer, **kwargs) -> List[ResultRow]:
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
        tid = repo.dataset_to_tid(dataset)
        for fold in range(n_eval_folds):
            dataset_fold_name = repo.task_name(tid=tid, fold=fold)
            automl_df_fold = automl_df[automl_df['task'] == dataset_fold_name]
            task_automl_dict = automl_df_fold.T.to_dict()

            for k, v in task_automl_dict.items():
                assert tid == v['tid']
                metric_error = v['metric_error']
                rows_automl.append(ResultRow(
                    taskid=tid,
                    fold=v['fold'],
                    method=v['framework'],
                    test_error=metric_error,
                    rank=rank_scorer.rank(dataset_fold_name, metric_error),
                    normalized_score=normalized_scorer.rank(dataset_fold_name, metric_error),
                ))

    return rows_automl


def zeroshot_name(
        n_portfolio: int = 20, n_ensemble: int = 40, n_training_dataset: int = None, n_training_fold: int = None,
        max_runtime: float = None
):
    """
    :return: name of the zeroshot method such as Zeroshot-N20-C40 if n_training_dataset or n_training_folds are not
    None, suffixes "-D{n_training_dataset}" and "-S{n_training_folds}" are added, for instance "Zeroshot-N20-C40-D30-S5"
    would be the name if n_training_dataset=30 and n_training_fold=5
    """
    suffix = [
        f"-{letter}{x}" if x is not None else ""
        for letter, x in [("N", n_portfolio), ("C", n_ensemble), ("D", n_training_dataset), ("S", n_training_fold), ("T", max_runtime)]
    ]
    suffix = "".join(suffix)
    return f"Zeroshot{suffix}"


def zeroshot_results(
        repo: EvaluationRepository,
        dataset_names: List[str],
        rank_scorer,
        normalized_scorer,
        n_eval_folds: int,
        n_ensembles: List[int] = [40],
        n_portfolios: List[int] = [20],
        n_training_datasets: List[int] = [None],
        n_training_folds: List[int] = [None],
        max_runtimes: List[float] = [None],
        engine: str = "ray",
) -> List[ResultRow]:
    """
    :param dataset_names: list of dataset to use when fitting zeroshot
    :param n_eval_folds: number of folds to consider for evaluation
    :param n_ensembles: number of caruana sizes to consider
    :param n_portfolios: number of folds to use when fitting zeroshot
    :param n_training_datasets: number of dataset to use when fitting zeroshot
    :param n_training_folds: number of folds to use when fitting zeroshot
    :param engine: engine to use, must be "sequential", "joblib" or "ray"
    :return: evaluation obtained on all combinations
    """
    def evaluate_dataset(test_dataset, portfolio_size, ensemble_size, n_training_dataset, n_training_fold, max_runtime, repo: EvaluationRepository, df_rank, rank_scorer, normalized_scorer):
        method_name = zeroshot_name(
            portfolio_size, ensemble_size, n_training_dataset,
            n_training_fold=n_training_fold, max_runtime=max_runtime
        )
        if n_training_fold is None:
            n_training_fold = n_eval_folds
        rows_zeroshot = []
        test_tid = repo.dataset_to_tid(test_dataset)
        available_tids = [repo.dataset_to_tid(dataset) for dataset in dataset_names if dataset != test_dataset]
        np.random.shuffle(available_tids)
        if n_training_dataset is None:
            n_training_dataset = len(available_tids)
        selected_tids = set(available_tids[:n_training_dataset])

        train_tasks = []
        for task in df_rank.columns:
            tid, fold = task.split("_")
            if int(tid) in selected_tids and int(fold) < n_training_fold:
                train_tasks.append(task)

        indices = zeroshot_configs(-df_rank[train_tasks].values.T, portfolio_size)
        portfolio_configs = [df_rank.index[i] for i in indices]
        portfolio_configs = sort_by_runtime(repo=repo, config_names=portfolio_configs)
        portfolio_configs = filter_configs_by_runtime(
            repo=repo,
            tid=test_tid,
            fold=0,
            config_names=portfolio_configs,
            max_cumruntime=max_runtime,
        )

        assert len(portfolio_configs) > 0
        test_errors = repo.evaluate_ensemble(
            tids=[test_tid],
            config_names=portfolio_configs,
            ensemble_size=ensemble_size,
            rank=False,
            backend='native',
        )
        assert test_errors.shape[0] == 1  # we send one model, we should get one row back
        for fold in range(n_eval_folds):
            test_error = test_errors[0][fold]
            dataset_fold_name = repo.task_name(tid=test_tid, fold=fold)
            rows_zeroshot.append(ResultRow(
                taskid=test_tid,
                fold=fold,
                method=method_name,
                test_error=test_error,
                rank=rank_scorer.rank(dataset_fold_name, test_error),
                normalized_score=normalized_scorer.rank(dataset_fold_name, test_error),
                config_selected=portfolio_configs,
            ))
        return rows_zeroshot

    dd = repo._zeroshot_context.df_results_by_dataset_vs_automl
    df_rank = dd.pivot_table(index="framework", columns="dataset", values="score_val").rank()
    df_rank.fillna(value=np.nanmax(df_rank.values), inplace=True)
    assert not any(df_rank.isna().values.reshape(-1))

    list_rows = parallel_for(
        evaluate_dataset,
        inputs=list(itertools.product(dataset_names, n_portfolios, n_ensembles, n_training_datasets, n_training_folds, max_runtimes)),
        context=dict(repo=repo, df_rank=df_rank, rank_scorer=rank_scorer, normalized_scorer=normalized_scorer),
        engine=engine,
    )
    return [x for l in list_rows for x in l]



def evaluate_tuning(
        repo: EvaluationRepository, dataset_names: List[str], n_eval_folds: int, rank_scorer, normalized_scorer,
        expname="02-05-v2",
        **kwargs,
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
        tid = repo.dataset_to_tid(dataset)
        for suffix, ensemble_size in [("", 1), (f" (ensemble)", 20)]:
            for method in ["zeroshot", "localsearch"]:
                test_errors = repo.evaluate_ensemble(
                    tids=[tid],
                    config_names=taskid_to_config(tuning_rows, tid)[method],
                    ensemble_size=ensemble_size,
                    rank=False,
                )
                assert test_errors.shape[0] == 1  # we send one model, we should get one row back
                for fold in range(n_eval_folds):
                    test_error = test_errors[0][fold]
                    task_name = repo.task_name(tid=tid, fold=fold)
                    rows.append(ResultRow(
                        taskid=tid,
                        fold=fold,
                        method=f"{method}{suffix}".capitalize(),
                        test_error=test_error,
                        rank=rank_scorer.rank(task_name, test_error),
                        normalized_score=normalized_scorer.rank(task_name, test_error),
                    ))
    return rows
