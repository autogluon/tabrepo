"""
Compares multiple baselines such as:
* AutoGluon with different presets
* Extensive search among all configurations (608 in the current example)
* Best of all LightGBM models
* Zeroshot portfolio of 10 configs
"""
import ast

import matplotlib.pyplot as plt
from matplotlib import cm
import string
import random
from pathlib import Path

from syne_tune.experiments import load_experiments_df
from tqdm import tqdm
import numpy as np
from autogluon_zeroshot.simulation.repository import load
import pandas as pd

from autogluon_zeroshot.utils import catchtime
from autogluon_zeroshot.utils.normalized_scorer import NormalizedScorer
from autogluon_zeroshot.utils.rank_utils import RankScorer
from dataclasses import dataclass
from typing import List

@dataclass
class MethodStyle:
    name: str
    color: str
    linestyle: str = "-"
    label: str = None


def random_string(length: int) -> str:
    pool = string.ascii_letters + string.digits
    return "".join(random.choice(pool) for _ in range(length))

def dataset_fold_name(repo, dataset, fold):
    return f"{repo.dataset_to_taskid(dataset)}_{fold}"

def get_automl_errors(repo, dataset, fold):
    return repo._zeroshot_context.rank_scorer_vs_automl.error_dict[dataset_fold_name(repo, dataset, fold)]

def evaluate_automl(repo, dataset_names, n_folds):
    rows_automl = []
    for dataset_name in dataset_names:
        for fold in range(n_folds):
            for method, test_error in get_automl_errors(repo, dataset_name, fold).to_dict().items():
                rows_automl.append({
                    "dataset": dataset_name,
                    "fold": fold,
                    "method": method,
                    "test-error": test_error
                })
    return rows_automl

def evaluate_ensemble(repo, dataset_name, fold, configs, ensemble_size: int = 20):
    return repo.evaluate_ensemble(
        dataset_names=[dataset_name],
        config_names=configs,
        folds=[fold],
        rank=False,
        ensemble_size=ensemble_size,
    ).reshape(-1)[0]

def evaluate_basemodels(repo, dataset_names, n_folds, frameworks, config_names):
    rows_basemodels = []
    for dataset_name in tqdm(dataset_names):
        for fold in range(n_folds):
            metrics = repo.eval_metrics(dataset_name=dataset_name, config_names=config_names, fold=fold,
                                        check_all_found=False)
            for metric in metrics:
                rows_basemodels.append({
                    "dataset": dataset_name,
                    "fold": fold,
                    "method": metric["framework"],
                    "test-error": metric["metric_error"],
                    "score_val": metric["score_val"],
                })

            # compute best of 10 configuration for all framework
            df_metric = pd.DataFrame(metrics).sort_values("score_val", ascending=False)
            for framework in frameworks:
                df_framework = df_metric[df_metric.framework.str.contains(framework)]
                if len(df_framework) > 0:
                    df_top = df_framework.head(10)
                    row = {
                        "dataset": dataset_name,
                        "fold": fold,
                        "method": f"Best of 10 {framework}",
                        "test-error": df_top['metric_error'].values[0],
                        # select configuration with the best validation error
                        "score_val": df_top['score_val'].values[0],
                    }
                    rows_basemodels.append(row)

                    # evaluate ensemble
                    configs = df_top.framework.tolist()
                    test_error = evaluate_ensemble(
                        repo=repo, dataset_name=dataset_name, configs=configs, fold=fold,
                    )
                    row = {
                        "dataset": dataset_name,
                        "fold": fold,
                        "method": f"Best of 10 {framework} (ensemble)",
                        "test-error": test_error,
                    }
                    rows_basemodels.append(row)

            # compute best of all framework
            df_top = df_metric.head(10)
            row = {
                "dataset": dataset_name,
                "fold": fold,
                "method": f"Best of 10 all framework",
                "test-error": df_top['metric_error'].values[0],  # select configuration with best validation error
                "score_val": df_top['score_val'].values[0],
            }
            rows_basemodels.append(row)

            # compute ensemble of all models
            configs = df_top.framework.tolist()
            test_error = evaluate_ensemble(
                repo=repo, dataset_name=dataset_name, configs=configs, fold=fold,
            )
            row = {
                "dataset": dataset_name,
                "fold": fold,
                "method": f"Best of 10 (ensemble)",
                "test-error": test_error,  # select configuration with best validation error
            }
            rows_basemodels.append(row)
    return rows_basemodels

def make_scorers(repo):
    df_results_baselines = pd.concat([
        repo._zeroshot_context.df_results_by_dataset_vs_automl,
        repo._zeroshot_context.df_results_by_dataset_automl,
    ], ignore_index=True)
    unique_dataset_folds = [dataset_fold_name(repo, dataset_name, fold) for dataset_name in repo.dataset_names() for fold in
                            range(repo.n_folds())]
    rank_scorer = RankScorer(df_results_baselines, datasets=unique_dataset_folds, pct=False)
    normalized_scorer = NormalizedScorer(df_results_baselines, datasets=unique_dataset_folds, baseline=None)
    return rank_scorer, normalized_scorer


def load_synetune_results(expname):
    from syne_tune.experiments import load_experiments_df
    import ast
    name_filter = lambda path: expname in str(path)
    df_results = load_experiments_df(path_filter=name_filter)
    df_results["fold"] = df_results.apply(lambda row: int(row['tuner_name'].split("fold-")[1].split("-")[0]), axis=1)
    for col in ['configs', 'train_datasets', 'test_datasets']:
        df_results[col] = df_results[f'config_{col}'].apply(lambda x: ast.literal_eval(x))
    return df_results


def evaluate_tuning(repo, dataset_names, n_folds, expname="02-05-v2"):
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
                    rows.append({
                        "dataset": dataset,
                        "fold": fold,
                        "method": f"{method}{suffix}",
                        "test-error": test_errors[0][fold]
                    })
    return rows

def evaluate_zeroshot(repo, dataset_names, n_folds, portfolio_sizes=[5, 10, 20, 40, 80]):
    from autogluon_zeroshot.portfolio.zeroshot_selection import zeroshot_configs
    dd = repo._zeroshot_context.df_results_by_dataset_vs_automl
    df_rank = dd.pivot_table(index="framework", columns="dataset", values="score_val").rank()
    df_rank.fillna(value=np.nanmax(df_rank.values), inplace=True)
    assert not any(df_rank.isna().values.reshape(-1))

    rows_zeroshot = []

    for dataset in tqdm(dataset_names):
        for portfolio_size in portfolio_sizes:
            taskid = repo.dataset_to_taskid(dataset)
            train_datasets = [x for x in df_rank.columns if x.split("_")[0] != str(taskid)]
            indices = zeroshot_configs(-df_rank[train_datasets].values.T, portfolio_size)
            portfolio_configs = [df_rank.index[i] for i in indices]

            # run best base model and ensemble
            for suffix, ensemble_size in [(f"-{portfolio_size}", 1), (f"-{portfolio_size} (ensemble)", 20)]:
                test_errors = repo.evaluate_ensemble(
                    dataset_names=[dataset],
                    config_names=portfolio_configs,
                    ensemble_size=ensemble_size,
                    rank=False,
                )
                assert test_errors.shape[0] == 1  # we send one model, we should get one row back
                for fold in range(n_folds):
                    row = {
                        "dataset": dataset,
                        "fold": fold,
                        "method": f"Zeroshot{suffix}",
                        "test-error": test_errors[0][fold]
                    }
                    rows_zeroshot.append(row)

    return rows_zeroshot


def generate_results(n_datasets, n_folds, frameworks):
    repo = load(version="BAG_D244_F10_C608_FULL")
    rank_scorer, normalized_scorer = make_scorers(repo)

    dataset_names = repo.dataset_names()
    config_names = repo.list_models_available(dataset_names[0])
    if n_datasets is not None:
        dataset_names = dataset_names[:n_datasets]

    print(f"Original number of datasets: {len(repo.dataset_names())}, kept only: {len(dataset_names)}")

    rows_automl = evaluate_automl(repo, dataset_names, n_folds)
    rows_basemodels = evaluate_basemodels(repo, dataset_names, n_folds, frameworks, config_names)
    rows_zeroshot = evaluate_zeroshot(repo, dataset_names, n_folds)
    df = pd.DataFrame(rows_automl + rows_basemodels + rows_zeroshot)
    df["rank"] = df.apply(
        lambda row: rank_scorer.rank(dataset_fold_name(repo, row["dataset"], row["fold"]), row["test-error"]),
        axis=1
    )
    df["normalized-score"] = df.apply(
        lambda row: normalized_scorer.rank(dataset_fold_name(repo, row["dataset"], row["fold"]), row["test-error"]),
        axis=1
    )
    return df


def load_cache_or_regenerate(name, ignore_cash, n_datasets, n_folds, frameworks):
    cache_file = Path(__file__).parent / f"results-{name}-{n_datasets}-{n_folds}.csv.zip"
    if not cache_file.exists() or (cache_file.exists() and ignore_cash):
        with catchtime(f"Evaluating model results and saving to {cache_file}"):
            df = generate_results(n_datasets=n_datasets, n_folds=n_folds, frameworks=frameworks)
            df.to_csv(cache_file, index=False)
    else:
        print(f"load results from {cache_file}")
        df = pd.read_csv(cache_file)
    return df


def show_latex_table(df):
    avg_metrics = {}
    for metric in ["rank", "normalized-score"]:
        avg_metric = df.groupby("method").mean(numeric_only=True)[metric]
        avg_metric.sort_values().head(60)

        avg_metric = avg_metric[
            avg_metric.index.str.contains("|".join(["AutoGluon", "Zeroshot", "LocalSearch", "Best of 10"]))]
        xx = avg_metric.sort_values()
        xx.index = [x.replace("_c1_BAG_L1", " default") for x in xx.index]
        avg_metrics[metric] = xx
    print(pd.DataFrame(avg_metrics).sort_values(by="rank").to_latex(float_format="%.2f"))

def show_cdf(df, method_styles: List[MethodStyle]):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
    metrics = ["normalized-score", "rank"]
    for i, metric in enumerate(metrics):
        for j, method_style in enumerate(method_styles):
            xx = df.loc[df.method == method_style.name, metric].sort_values()
            if len(xx) > 0:
                axes[i].plot(
                    xx.values, np.arange(len(xx)) / len(xx),
                    label=method_style.label,
                    color=method_style.color,
                    linestyle=method_style.linestyle,
                    lw=1.5,
                )
                axes[i].set_title(f"{metric}")
                axes[i].set_xlabel(f"{metric}")
                if i == 0:
                    axes[i].set_ylabel(f"CDF")
            else:
                print(f"Could not find method {method_style.name}")
    axes[-1].legend()
    plt.tight_layout()
    fig_save_path = Path(__file__).parent / "figures" / "cdf.pdf"
    fig_save_path_dir = fig_save_path.parent
    fig_save_path_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_save_path)
    plt.show()


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--n_folds", type=int, default=2)
    parser.add_argument("--n_datasets", type=int, default=2)
    parser.add_argument("--ignore_cash", type=int, default=0)
    parser.add_argument("--name", type=str, help="Name of the experiment", default=random_string(5))

    args, _ = parser.parse_known_args()
    print(args.__dict__)

    n_folds = args.n_folds
    n_datasets = args.n_datasets
    ignore_cash = args.ignore_cash
    name = args.name

    frameworks = ["CatBoost", "ExtraTrees", "LightGBM", "NeuralNetFastAI", "RandomForest"]
    df = load_cache_or_regenerate(name, ignore_cash, n_datasets, n_folds, frameworks)
    rename_dict = {
        "AutoGluon_bq_1h8c_2023_03_19_zs": "AutoGluon best quality (ensemble)",
        "AutoGluon_bq_1h8c_2023_03_19_zs_autogluon_single": "AutoGluon best quality",
        "AutoGluon_hq_1h8c_2023_03_19_zs": "AutoGluon high quality (ensemble)",
        "Best of 10 (ensemble)": "Best of 10 frameworks (ensemble)",
        "Best of 10 all framework": "Best of 10 frameworks",
        "AutoGluon_mq_1h8c_2023_03_19_zs": "AutoGluon medium quality (ensemble)",
        "AutoGluon_mq_1h8c_2023_03_19_zs_autogluon_single": "AutoGluon medium quality",
        "AutoGluon_mq_1h8c_2023_03_19_zs_LightGBM": "AutoGluon medium quality only LightGBM",
    }
    df["method"] = df["method"].replace(rename_dict)

    methods_to_show = [
        "AutoGluon best quality",
        "AutoGluon high quality",
        "AutoGluon medium quality",
        "Best of 10 frameworks",
        "Zeroshot",
        "LocalSearch",
        "Best of 10 LightGBM"
    ]

    methods_to_show += [x + " (ensemble)" for x in methods_to_show]

    df = df[df.method.isin(methods_to_show)]

    # 1) Show latex table of average results
    show_latex_table(df)

    method_styles = [
        MethodStyle(f"AutoGluon {preset} quality", color=cm.get_cmap("tab20c")(i), linestyle="--")
        for i, preset in enumerate(["best", "high", "medium"])
    ] + [
        MethodStyle("Best of 10 frameworks", color="red", linestyle="--"),
        MethodStyle("Zeroshot", color="black", linestyle="--"),
        MethodStyle("Best of 10 LightGBM", color="orange", linestyle="--")
    ]
    # add ensemble styles
    method_styles += [
        MethodStyle(m.name + " (ensemble)", color=m.color, linestyle="-", label=m.name)
        for m in method_styles
    ]

    # 2) show cdf
    show_cdf(df, method_styles)

if __name__ == '__main__':
    main()