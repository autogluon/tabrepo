"""
Compares multiple baselines such as:
* AutoGluon with different presets
* Extensive search among all configurations (608 in the current example)
* Best of all LightGBM models
* Zeroshot portfolio of 10 configs
"""
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import List, Callable
import numpy as np
import pandas as pd
from pathlib import Path
from autogluon_zeroshot.simulation.repository import load
from autogluon_zeroshot.utils.cache import cache_function, cache_function_dataframe
from autogluon_zeroshot.utils.normalized_scorer import NormalizedScorer
from autogluon_zeroshot.utils.rank_utils import RankScorer
from dataclasses import dataclass

from scripts.baseline_comparison.baselines import zeroshot_results, automl_results, ResultRow, evaluate_tuning
from scripts.baseline_comparison.plot_utils import show_latex_table, show_cdf, MethodStyle


@dataclass
class Experiment:
    expname: str  # name of the parent experiment used to store the file
    name: str  # name of the specific experiment, e.g. "localsearch"
    run_fun: Callable[[], List[ResultRow]]  # function to execute to obtain results

    def data(self, ignore_cache: bool = False):
        return cache_function_dataframe(
            lambda: pd.DataFrame(self.run_fun()),
            cache_name=self.name,
            ignore_cache=ignore_cache,
            cache_path=Path(__file__).parent.parent.parent / "data" / "results-baseline-comparison" / self.expname,
        )

def make_scorers(repo):
    df_results_baselines = pd.concat([
        repo._zeroshot_context.df_results_by_dataset_vs_automl,
        repo._zeroshot_context.df_results_by_dataset_automl,
    ], ignore_index=True)
    unique_dataset_folds = [
        f"{repo.dataset_to_taskid(dataset)}_{fold}"
        for dataset in repo.dataset_names()
        for fold in range(repo.n_folds())
    ]
    rank_scorer = RankScorer(df_results_baselines, datasets=unique_dataset_folds, pct=False)
    normalized_scorer = NormalizedScorer(df_results_baselines, datasets=unique_dataset_folds, baseline=None)
    return rank_scorer, normalized_scorer

def list_artificial_experiments() -> List[Experiment]:
    # a list of artificial experiments which allow to quickly check the results
    expname = "artificial"
    datasets = [f"d{i}" for i in range(10)]
    folds = list(range(3))
    return [
        Experiment(expname, "method-group-1", lambda : [ResultRow(
            dataset=dataset,
            fold=fold,
            method=method,
            test_error=np.random.rand(),
            rank=np.random.rand(),
            normalized_score=np.random.rand(),
        ) for dataset in datasets for fold in folds for method in ["dummy1", "dummy2"]
        ])
    ]


def list_experiments(n_datasets: int, n_folds: int, expname: str, repo_version: str = "BAG_D244_F10_C608_FULL") -> List[Experiment]:
    repo = cache_function(lambda: load(version=repo_version), cache_name="repo")
    rank_scorer, normalized_scorer = make_scorers(repo)
    dataset_names = repo.dataset_names()
    if n_datasets is not None:
        dataset_names = dataset_names[:n_datasets]
    if n_folds is None:
        n_folds = repo.n_folds()
    experiment_common_kwargs = dict(
        repo=repo,
        dataset_names=dataset_names,
        n_folds=n_folds,
        rank_scorer=rank_scorer,
        normalized_scorer=normalized_scorer
    )
    return [
        # Automl baselines such as Autogluon best, high, medium quality
        Experiment(
            expname=expname, name=f"automl-baselines-{expname}",
            run_fun=lambda: automl_results(**experiment_common_kwargs),
        ),
        Experiment(
            expname=expname, name=f"zeroshot-multiple-portfolio-sizes-{expname}",
            run_fun=lambda: zeroshot_results(**experiment_common_kwargs),
        ),
        Experiment(
            expname=expname, name=f"zeroshot-multiple-caruana-sizes-{expname}",
            run_fun=lambda : zeroshot_results(
                ensemble_sizes=[20, 40, 80], portfolio_sizes=[20, 40], **experiment_common_kwargs
            )
        ),
        # Results of local search, results would only be reported if `run_method_comparison.py` was evaluted with
        # the specified `expname`
        Experiment(
            expname=expname, name="localsearch",
            run_fun=lambda: evaluate_tuning(**experiment_common_kwargs, expname="02-05-v2")
        )
    ]


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--n_folds", type=int, required=False, help="Number of datasets to consider when evaluating all baselines.")
    parser.add_argument("--n_datasets", type=int, required=False, help="Number of datasets to consider when evaluating all baselines.")
    parser.add_argument("--ignore_cache", action="store_true", help="Ignore previously generated results and recompute them from scratch.")
    parser.add_argument("--quick_check", action="store_true", help="Perform a quick check of the script with artificial data")
    parser.add_argument("--expname", type=str, help="Name of the experiment", default="dummy")
    args = parser.parse_args()
    print(args.__dict__)

    n_folds = args.n_folds
    n_datasets = args.n_datasets
    quick_check = args.quick_check
    ignore_cache = args.ignore_cache

    if quick_check:
        experiments = list_artificial_experiments()
        df = pd.concat([
            experiment.data(ignore_cache=ignore_cache) for experiment in experiments
        ])

        method_styles = None
    else:
        expname = args.expname

        expname += f"-folds_{n_folds}" if n_folds else f"-folds_all"
        expname += f"-datasets_{n_datasets}" if n_datasets else f"-datasets_all"
        experiments = list_experiments(
            expname=expname, n_folds=n_folds, n_datasets=n_datasets
        )
        method_styles = [
            MethodStyle(f"Zeroshot-{n_portfolio}-{20} (ensemble)", color=cm.get_cmap("viridis")(i / 4), linestyle="-")
            for i, n_portfolio in enumerate([5, 10, 20, 40, 80])
        ] + [
            MethodStyle("AutoGluon best quality (ensemble)", color="black", linestyle="--"),
            # MethodStyle("AutoGluon best quality", color="black", linestyle="dashdot"),
            MethodStyle("AutoGluon high quality (ensemble)", color="black", linestyle=":"),
            MethodStyle("Localsearch (ensemble)", color="red"),
            # MethodStyle("localsearch (ensemble) (ST)", color="red", linestyle="-")
        ]
        df = pd.concat([
            experiment.data(ignore_cache=ignore_cache) for experiment in experiments
        ])
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

    print(f"Obtained {len(df)} evaluations on {len(df.dataset.unique())} datasets for {len(df.method.unique())} methods.")
    print(f"Methods available: {sorted(df.method.unique())}.")
    if method_styles:
        for method_style in method_styles:
            if not method_style.name in df.method.unique():
                print(f"Method style {method_style.name} not found in evaluations.")
    show_latex_table(df)
    show_cdf(df, method_styles=method_styles)
    plt.show()
