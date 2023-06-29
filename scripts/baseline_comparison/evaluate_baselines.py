from typing import List, Callable
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from matplotlib import cm

from autogluon_zeroshot.repository.evaluation_repository import load
from autogluon_zeroshot.utils.cache import cache_function, cache_function_dataframe
from scripts.baseline_comparison.baselines import automl_results, framework_default_results, \
    framework_best_results, zeroshot_results, zeroshot_name, ResultRow
from scripts.baseline_comparison.plot_utils import MethodStyle, show_latex_table, show_cdf
from autogluon_zeroshot.utils.normalized_scorer import NormalizedScorer
from autogluon_zeroshot.utils.rank_utils import RankScorer
from dataclasses import dataclass


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
        f"{repo.dataset_to_tid(dataset)}_{fold}"
        for dataset in repo.dataset_names()
        for fold in range(repo.n_folds())
    ]
    rank_scorer = RankScorer(df_results_baselines, datasets=unique_dataset_folds, pct=False)
    normalized_scorer = NormalizedScorer(df_results_baselines, datasets=unique_dataset_folds, baseline=None)
    return rank_scorer, normalized_scorer


def impute_missing(repo):
    # impute random forest data missing folds by picking data from another fold
    # TODO remove once we have complete dataset
    df = repo._zeroshot_context.df_results_by_dataset_vs_automl
    df["framework_type"] = df.apply(lambda row: row["framework"].split("_")[0], axis=1)

    missing_tasks = [(3583, 0), (58, 9), (3483, 0)]
    for tid, fold in missing_tasks:
        impute_fold = (fold + 1) % 10
        df_impute = df[(df.framework_type == 'RandomForest') & (df.dataset == f"{tid}_{impute_fold}")].copy()
        df_impute['dataset'] = f"{tid}_{fold}"
        df_impute['fold'] = fold
        df = pd.concat([df, df_impute], ignore_index=True)
    repo._zeroshot_context.df_results_by_dataset_vs_automl = df


def plot_figure(df, method_styles: List[MethodStyle], title: str = None, figname: str = None):
    fig, _ = show_cdf(df[df.method.isin([m.name for m in method_styles])], method_styles=method_styles)
    if title:
        fig.suptitle(title)
    if figname:
        fig_save_path = Path(__file__).parent.parent / "figures" / f"{figname}.pdf"
        fig_save_path_dir = fig_save_path.parent
        fig_save_path_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_save_path)
    plt.show()



if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--n_folds", type=int, default=10, required=False, help="Number of folds to consider when evaluating all baselines.")
    parser.add_argument("--n_datasets", type=int, required=False, help="Number of datasets to consider when evaluating all baselines.")
    parser.add_argument("--ignore_cache", action="store_true", help="Ignore previously generated results and recompute them from scratch.")
    parser.add_argument("--expname", type=str, help="Name of the experiment", default="dummy")
    parser.add_argument("--engine", type=str, required=False, default="ray", choices=["sequential", "ray", "joblib"],
                        help="Engine used for embarrassingly parallel loop.")
    args = parser.parse_args()
    print(args.__dict__)

    ignore_cache = args.ignore_cache
    engine = args.engine
    expname = args.expname
    n_datasets = args.n_datasets
    if n_datasets:
        expname += f"-{n_datasets}"

    n_eval_folds = args.n_folds
    n_portfolios = [5, 10, 20, 40, 80]
    # n_ensembles=[5, 10, 20, 40, 80]
    max_runtimes = [60, 120, 240, 3600]
    n_training_datasets = [1, 4, 16, 32, 64, 130]
    n_training_folds = [1, 2, 5, 10]
    n_training_configs = [1, 2, 5, 50, 100]
    n_ensembles = [10, 20, 40, 80]
    linestyle_ensemble = "--"
    linestyle_default = "dotted"
    linestyle_tune = "-"


    repo = cache_function(lambda: load(version="BAG_D244_F10_C608_FULL"), cache_name="repo")
    rank_scorer, normalized_scorer = make_scorers(repo)
    missing_tids = [359932, 359944, 359933, 359946]
    dataset_names = [ds for ds in repo.dataset_names() if not repo.dataset_to_tid(ds) in missing_tids]
    if n_datasets:
        dataset_names = dataset_names[:n_datasets]
    impute_missing(repo)

    experiment_common_kwargs = dict(
        repo=repo,
        dataset_names=dataset_names,
        rank_scorer=rank_scorer,
        normalized_scorer=normalized_scorer,
        n_eval_folds=n_eval_folds,
    )

    experiments = [
        Experiment(
            expname=expname, name=f"framework-default-{expname}",
            run_fun=lambda: framework_default_results(**experiment_common_kwargs)
        ),
        Experiment(
            expname=expname, name=f"framework-best-{expname}",
            run_fun=lambda: framework_best_results(**experiment_common_kwargs),
        ),
        Experiment(
            expname=expname, name=f"framework-all-best-{expname}",
            run_fun=lambda: framework_best_results(framework_types=[None], n_configs=[10, 20, 100, 608], **experiment_common_kwargs),
        ),
        # Automl baselines such as Autogluon best, high, medium quality
        Experiment(
            expname=expname, name=f"automl-baselines-{expname}",
            run_fun=lambda: automl_results(**experiment_common_kwargs),
        ),
        Experiment(
            expname=expname, name=f"zeroshot-{expname}",
            run_fun=lambda: zeroshot_results(**experiment_common_kwargs)
        ),
        Experiment(
            expname=expname, name=f"zeroshot-{expname}-maxruntimes",
            run_fun=lambda: zeroshot_results(max_runtimes=max_runtimes, **experiment_common_kwargs)
        ),
        Experiment(
            expname=expname, name=f"zeroshot-{expname}-num-training-datasets",
            run_fun=lambda: zeroshot_results(n_training_datasets=n_training_datasets, **experiment_common_kwargs)
        ),
        # Experiment(
        #     expname=expname, name=f"zeroshot-{expname}-num-folds",
        #     run_fun=lambda: zeroshot_results(n_training_folds=n_training_folds, ** experiment_common_kwargs)
        # ),
        Experiment(
            expname=expname, name=f"zeroshot-{expname}-num-portfolios",
            run_fun=lambda: zeroshot_results(n_portfolios=n_portfolios, **experiment_common_kwargs)
        ),
        Experiment(
            expname=expname, name=f"zeroshot-{expname}-num-configs",
            run_fun=lambda: zeroshot_results(n_training_configs=n_training_configs, **experiment_common_kwargs)
        ),
        Experiment(
            expname=expname, name=f"zeroshot-{expname}-num-caruana",
            run_fun=lambda: zeroshot_results(n_ensembles=n_ensembles, **experiment_common_kwargs)
        ),

    ]

    df = pd.concat([
        experiment.data(ignore_cache=ignore_cache) for experiment in experiments
    ])
    rename_dict = {
        "AutoGluon_bq_1h8c_2023_03_19_zs": "AutoGluon best quality (ensemble)",
        "AutoGluon_bq_1h8c_2023_03_19_zs_autogluon_single": "AutoGluon best quality",
        "AutoGluon_hq_1h8c_2023_03_19_zs": "AutoGluon high quality (ensemble)",
        # "Best of 10 (ensemble)": "Best of 10 frameworks (ensemble)",
        # "Best of 10 all framework": "Best of 10 frameworks",
        "AutoGluon_mq_1h8c_2023_03_19_zs": "AutoGluon medium quality (ensemble)",
        "AutoGluon_mq_1h8c_2023_03_19_zs_autogluon_single": "AutoGluon medium quality",
        "AutoGluon_mq_1h8c_2023_03_19_zs_LightGBM": "AutoGluon medium quality only LightGBM",
    }
    df["method"] = df["method"].replace(rename_dict)
    print(f"Obtained {len(df)} evaluations on {len(df.taskid.unique())} datasets for {len(df.method.unique())} methods.")
    print(f"Methods available:" + "\n".join(sorted(df.method.unique())))
    print("all")
    show_latex_table(df)#, ["rank", "normalized_score", ])

    ag_styles = [
        MethodStyle("AutoGluon best quality (ensemble)", color="black", linestyle="--", label_str="AG-best"),
        # MethodStyle("AutoGluon high quality (ensemble)", color="black", linestyle=":", label_str="AG-high"),
        # MethodStyle("localsearch (ensemble) (ST)", color="red", linestyle="-")
    ]

    method_styles = ag_styles.copy()
    frameworks = ["CatBoost", "NeuralNetFastAI", "LightGBM", "RandomForest", "ExtraTrees"]

    for i, framework_type in enumerate(frameworks):
        method_styles.append(
            MethodStyle(
                f"{framework_type} (default)",
                color=sns.color_palette('bright')[i],
                linestyle="dotted",
                label=False,
            )
        )
        method_styles.append(
            MethodStyle(
                f"{framework_type} (100 samples)",
                color=sns.color_palette('bright')[i],
                linestyle=linestyle_tune,
                label_str=framework_type,
                label=True,
            )
        )
        method_styles.append(
            MethodStyle(
                f"{framework_type} (100 samples + ensemble)",
                color=sns.color_palette('bright')[i],
                linestyle=linestyle_ensemble,
                label=False,
            )
        )
    method_styles.append(
        MethodStyle(
            f"All (100 samples)",
            color=sns.color_palette('bright')[len(frameworks)],
            linestyle=linestyle_tune,
            label=True,
            label_str="All"
        ))
    method_styles.append(
        MethodStyle(
            f"All (100 samples + ensemble)",
            color=sns.color_palette('bright')[len(frameworks)],
            linestyle=linestyle_ensemble,
            label=False,
        ))
    show_latex_table(df[df.method.isin([m.name for m in method_styles])])#, ["rank", "normalized_score", ])
    plot_figure(
        df, method_styles, figname="cdf-frameworks",
        # title="Comparison of frameworks",
    )

    # Plot effect number of training datasets
    method_styles = ag_styles + [
        MethodStyle(
            zeroshot_name(n_training_dataset=size),
            color=cm.get_cmap("viridis")(i / (len(n_training_datasets) - 1)), linestyle="-", label_str=f"ZS-D{size}",
        )
        for i, size in enumerate(n_training_datasets)
    ]
    plot_figure(df, method_styles, title="Effect of number of training datasets", figname="cdf-n-training-datasets")

    # # Plot effect number of training fold
    # method_styles = ag_styles + [
    #     MethodStyle(
    #         zeroshot_name(n_training_fold=size),
    #         color=cm.get_cmap("viridis")(i / (len(n_training_folds) - 1)), linestyle="-", label_str=f"ZS-S{size}",
    #     )
    #     for i, size in enumerate(n_training_folds)
    # ]
    # plot_figure(df, method_styles, title="Effect of number of training folds", figname="cdf-n-training-folds")

    # Plot effect number of portfolio size
    method_styles = ag_styles + [
        MethodStyle(
            zeroshot_name(n_portfolio=size),
            color=cm.get_cmap("viridis")(i / (len(n_portfolios) - 1)), linestyle="-", label_str=f"ZS-N{size}",
        )
        for i, size in enumerate(n_portfolios)
    ]
    plot_figure(df, method_styles, title="Effect of number of portfolio configurations", figname="cdf-n-configs")

    # Plot effect of number of training configurations
    method_styles = ag_styles + [
        MethodStyle(
            zeroshot_name(n_training_config=size),
            color=cm.get_cmap("viridis")(i / (len(n_training_configs) - 1)), linestyle="-", label_str=f"ZS-M{size}",
        )
        for i, size in enumerate(n_training_configs)
    ]
    plot_figure(df, method_styles, title="Effect of number of offline configurations", figname="cdf-n-training-configs")

    # Plot effect of training time limit
    method_styles = ag_styles + [
        MethodStyle(
            zeroshot_name(max_runtime=size),
            color=cm.get_cmap("viridis")(i / (len(max_runtimes) - 1)), linestyle="-", label_str=f"ZS-T{size}",
        )
        for i, size in enumerate(max_runtimes)
    ]
    plot_figure(df, method_styles, title="Effect of training time limit", figname="cdf-max-runtime")

    # Plot effect of training time limit
    method_styles = ag_styles + [
        MethodStyle(
            zeroshot_name(n_ensemble=size),
            color=cm.get_cmap("viridis")(i / (len(n_ensembles) - 1)), linestyle="-", label_str=f"ZS-C{size}",
        )
        for i, size in enumerate(n_ensembles)
    ]
    plot_figure(df, method_styles, title="Effect of number of Caruana steps", figname="cdf-caruana")

    show_latex_table(df[(df.method.str.contains("Zeroshot") | (df.method.str.contains("AutoGluon best quality")))])
