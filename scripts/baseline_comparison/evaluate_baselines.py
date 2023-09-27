import os
import math
from typing import List, Callable, Dict
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from pathlib import Path

from autogluon_zeroshot.repository.evaluation_repository import (
    load,
    EvaluationRepository,
)
from autogluon_zeroshot.utils.cache import cache_function, cache_function_dataframe
from scripts.baseline_comparison.baselines import (
    automl_results,
    framework_default_results,
    framework_best_results,
    zeroshot_results,
    zeroshot_name,
    ResultRow,
    framework_types, framework_name, time_suffix, default_ensemble_size, n_portfolios_default,
)
from scripts.baseline_comparison.plot_utils import (
    MethodStyle,
    show_latex_table,
    show_cdf,
    show_scatter_performance_vs_time, iqm, show_scatter_performance_vs_time_lower_budgets, figure_path,
    plot_critical_diagrams,
)
from autogluon_zeroshot.utils.normalized_scorer import NormalizedScorer
from autogluon_zeroshot.utils.rank_utils import RankScorer
from dataclasses import dataclass
from scripts import output_path, load_context


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
            cache_path=output_path.parent / "data" / "results-baseline-comparison" / self.expname,
        )


def make_scorers(repo: EvaluationRepository):
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


def impute_missing(repo: EvaluationRepository):
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


def plot_figure(df, method_styles: List[MethodStyle], title: str = None, figname: str = None, show: bool = False):
    fig, _ = show_cdf(df[df.method.isin([m.name for m in method_styles])], method_styles=method_styles)
    if title:
        fig.suptitle(title)
    if figname:
        fig_save_path = figure_path() / f"{figname}.pdf"
        plt.tight_layout()
        plt.savefig(fig_save_path)
    if show:
        plt.show()


def make_rename_dict(suffix: str) -> Dict[str, str]:
    # return renaming of methods
    rename_dict = {}
    for hour in [1, 4, 24]:
        for automl_framework in ["autosklearn", "autosklearn2", "flaml", "lightautoml", "H2OAutoML"]:
            rename_dict[f"{automl_framework}_{hour}h{suffix}"] = f"{automl_framework} ({hour}h)".capitalize()
        for preset in ["best", "high", "medium"]:
            rename_dict[f"AutoGluon_{preset[0]}q_{hour}h{suffix}"] = f"AutoGluon {preset} ({hour}h)"
    for minute in [5, 10, 30]:
        for preset in ["best"]:
            rename_dict[f"AutoGluon_{preset[0]}q_{minute}m{suffix}"] = f"AutoGluon {preset} ({minute}m)"
    return rename_dict


def time_cutoff_baseline(df, rel_tol = 0.1):
    df = df.copy()
    # TODO Portfolio excess are due to just using one fold to simulate runtimes, fix it
    mask = (df["time fit (s)"] > df["fit budget"] * (1 + rel_tol)) & (~df.method.str.contains("Portfolio"))

    # gets performance of Extra-trees baseline on all tasks
    dd = repo._zeroshot_context.df_results_by_dataset_vs_automl
    dd = dd[dd.framework == "ExtraTrees_c1_BAG_L1"]
    dd["tid"] = dd.dataset.apply(lambda s: int(s.split("_")[0]))
    dd["fold"] = dd.dataset.apply(lambda s: int(s.split("_")[1]))
    dd["rank"] = dd.apply(lambda row: rank_scorer.rank(dataset=row["dataset"], error=row["metric_error"]), axis=1)
    dd["normalized-score"] = dd.apply(
        lambda row: normalized_scorer.rank(dataset=row["dataset"], error=row["metric_error"]), axis=1)
    df_baseline = dd[["tid", "fold", "rank", "normalized-score"]]

    df.loc[mask, ["normalized_score", "rank"]] = df.loc[mask, ["tid", "fold"]].merge(df_baseline, on=["tid", "fold"])[
        ["normalized-score", "rank"]].values

    return df


def rename_dataframe(df):
    rename_dict = make_rename_dict(suffix="8c_2023_08_21")
    df["method"] = df["method"].replace(rename_dict)
    df.rename({
        "normalized_score": "normalized-error",
        "time_train_s": "time fit (s)",
        "time_infer_s": "time infer (s)",
    },
        inplace=True, axis=1
    )

    def convert_timestamp(s):
        if "h)" in s:
            return float(s.split("(")[-1].replace("h)", "")) * 3600
        elif "m)" in s:
            return float(s.split("(")[-1].replace("m)", "")) * 60
        else:
            return None

    df["fit budget"] = df.method.apply(convert_timestamp)

    return df

def generate_sentitivity_plots(df, show: bool = False):
    # show stds

    # show stds
    fig, axes = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(10, 6))

    dimensions = [
        ("M", "Number of configuration per family"),
        ("D", "Number of training datasets to fit portfolios"),
    ]
    for i, (dimension, legend) in enumerate(dimensions):

        for j, metric in enumerate(["normalized-error", "rank"]):
            df_portfolio = df.loc[df.method.str.contains(f"Portfolio-N.*-{dimension}.*4h"), :].copy()
            df_ag = df.loc[df.method.str.contains("AutoGluon best \(4h\)"), metric].copy()

            df_portfolio.loc[:, dimension] = df_portfolio.loc[:, "method"].apply(lambda s: int(s.replace(" (ensemble) (4h)", "").split("-")[-1][1:]))

            dim, mean, sem = df_portfolio.loc[:, [dimension, metric]].groupby(dimension).agg(
                ["mean", "sem"]).reset_index().values.T
            mean = df_portfolio.loc[:, [dimension, metric]].groupby(dimension).agg(
                iqm).reset_index().loc[:, metric].values
            ax = axes[j][i]
            ax.errorbar(
                dim, mean, sem,
                label="Portfolio",
                linestyle="",
                marker="o",
            )
            ax.set_xlim([0, None])
            if j == 1:
                ax.set_xlabel(legend)
            if i == 0:
                ax.set_ylabel(f"Avg {metric}")
            ax.grid()
            ax.hlines(iqm(df_ag), xmin=min(dim), xmax=max(dim), color="black", label="AutoGluon", ls="--")
            if j == 1:
                ax.legend()
    fig_save_path = figure_path() / f"sensitivity.pdf"
    plt.tight_layout()
    plt.savefig(fig_save_path)
    plt.show()


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--repo", type=str, help="Name of the repo to load", default="BAG_D244_F3_C1416")
    parser.add_argument("--n_folds", type=int, default=-1, required=False,
                        help="Number of folds to consider when evaluating all baselines. Uses all if set to -1.")
    parser.add_argument("--n_datasets", type=int, required=False, help="Number of datasets to consider when evaluating all baselines.")
    parser.add_argument("--ignore_cache", action="store_true", help="Ignore previously generated results and recompute them from scratch.")
    parser.add_argument("--expname", type=str, help="Name of the experiment", default="dummy")
    parser.add_argument("--engine", type=str, required=False, default="ray", choices=["sequential", "ray", "joblib"],
                        help="Engine used for embarrassingly parallel loop.")
    parser.add_argument("--ray_process_ratio", type=float,
                        help="The ratio of ray processes to logical cpu cores. Use lower values to reduce memory usage. Only used if engine == 'ray'",)
    args = parser.parse_args()
    print(args.__dict__)

    repo_version = args.repo
    ignore_cache = args.ignore_cache
    ray_process_ratio = args.ray_process_ratio
    engine = args.engine
    expname = args.expname
    n_datasets = args.n_datasets
    if n_datasets:
        expname += f"-{n_datasets}"

    if engine == "ray" and args.ray_process_ratio is not None:
        assert (ray_process_ratio <= 1) and (ray_process_ratio > 0)
        num_cpus = os.cpu_count()
        num_ray_processes = math.ceil(num_cpus*ray_process_ratio)

        print(f'NOTE: To avoid OOM, we are limiting ray processes to {num_ray_processes} (Total Logical Cores: {num_cpus})\n'
              f'\tThis is based on ray_process_ratio={ray_process_ratio}')

        # FIXME: The large-scale 3-fold 244-dataset 1416-config runs OOM on m6i.32x without this limit
        import ray
        if not ray.is_initialized():
            ray.init(num_cpus=num_ray_processes)

    n_eval_folds = args.n_folds
    n_portfolios = [5, 10, 50, 100, n_portfolios_default]
    max_runtimes = [300, 600, 1800, 3600, 3600 * 4, 24 * 3600]
    # n_training_datasets = list(range(10, 210, 10))
    # n_training_configs = list(range(10, 210, 10))
    n_training_datasets = [5, 10, 50, 100, 150, 200]
    n_training_configs = [5, 10, 50, 100, 150, 200]
    n_training_folds = [1, 2, 5, 10]
    n_ensembles = [10, 20, 40, 80]
    linestyle_ensemble = "--"
    linestyle_default = "-"
    linestyle_tune = "dotted"

    # Number of digits to show in table
    n_digits = {
        "normalized-error": 3,
        "rank": 1,
        "time fit (s)": 1,
        "time infer (s)": 3,
    }


    repo: EvaluationRepository = load_context(version=repo_version)
    if n_eval_folds == -1:
        n_eval_folds = repo.n_folds()

    rank_scorer, normalized_scorer = make_scorers(repo)
    dataset_names = repo.dataset_names()
    if n_datasets:
        dataset_names = dataset_names[:n_datasets]

    experiment_common_kwargs = dict(
        repo=repo,
        dataset_names=dataset_names,
        rank_scorer=rank_scorer,
        normalized_scorer=normalized_scorer,
        n_eval_folds=n_eval_folds,
        engine=engine,
    )

    experiments = [
        Experiment(
            expname=expname, name=f"framework-default-{expname}",
            run_fun=lambda: framework_default_results(**experiment_common_kwargs)
        ),
        Experiment(
            expname=expname, name=f"framework-best-{expname}",
            run_fun=lambda: framework_best_results(max_runtimes=[3600, 3600 * 4, 3600 * 24], **experiment_common_kwargs),
        ),
        # Experiment(
        #     expname=expname, name=f"framework-all-best-{expname}",
        #     run_fun=lambda: framework_best_results(framework_types=[None], max_runtimes=[3600, 3600 * 4, 3600 * 24], **experiment_common_kwargs),
        # ),
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
        # Experiment(
        #     expname=expname, name=f"zeroshot-{expname}-num-folds",
        #     run_fun=lambda: zeroshot_results(n_training_folds=n_training_folds, ** experiment_common_kwargs)
        # ),
        Experiment(
            expname=expname, name=f"zeroshot-{expname}-num-portfolios",
            run_fun=lambda: zeroshot_results(n_portfolios=n_portfolios, n_ensembles=[1, default_ensemble_size], **experiment_common_kwargs)
        ),
    ]

    # Use more seeds
    for seed in range(10):
        experiments.append(Experiment(
            expname=expname, name=f"zeroshot-{expname}-num-configs-{seed}",
            run_fun=lambda: zeroshot_results(n_training_configs=n_training_configs, **experiment_common_kwargs)
        ))

        experiments.append(Experiment(
            expname=expname, name=f"zeroshot-{expname}-num-training-datasets-{seed}",
            run_fun=lambda: zeroshot_results(n_training_configs=n_training_configs, **experiment_common_kwargs)
        ))


    df = pd.concat([
        experiment.data(ignore_cache=ignore_cache) for experiment in experiments
    ])

    df = rename_dataframe(df)

    # df = time_cutoff_baseline(df)

    print(f"Obtained {len(df)} evaluations on {len(df.tid.unique())} datasets for {len(df.method.unique())} methods.")
    print(f"Methods available:" + "\n".join(sorted(df.method.unique())))
    total_time_h = df.loc[:, "time fit (s)"].sum() / 3600
    print(f"Total time of experiments: {total_time_h} hours")

    generate_sentitivity_plots(df, show=True)

    show_latex_table(df, "all", show_table=True, n_digits=n_digits)
    ag_styles = [
        # MethodStyle("AutoGluon best (1h)", color="black", linestyle="--", label_str="AG best (1h)"),
        MethodStyle("AutoGluon best (4h)", color="black", linestyle="-.", label_str="AG best (4h)", linewidth=2.5),
        # MethodStyle("AutoGluon high quality (ensemble)", color="black", linestyle=":", label_str="AG-high"),
        # MethodStyle("localsearch (ensemble) (ST)", color="red", linestyle="-")
    ]

    method_styles = ag_styles.copy()
    for i, framework_type in enumerate(framework_types):
        method_styles.append(
            MethodStyle(
                framework_name(framework_type, tuned=False),
                color=sns.color_palette('bright')[i],
                linestyle=linestyle_default,
                label=True,
                label_str=framework_type,
            )
        )
        method_styles.append(
            MethodStyle(
                framework_name(framework_type, max_runtime=4 * 3600, ensemble_size=1, tuned=True),
                color=sns.color_palette('bright')[i],
                linestyle=linestyle_tune,
                label=False,
            )
        )
        method_styles.append(
            MethodStyle(
                framework_name(framework_type, max_runtime=4 * 3600, tuned=True),
                color=sns.color_palette('bright')[i],
                linestyle=linestyle_ensemble,
                label=False,
                label_str=framework_type
            )
        )
    show_latex_table(df[df.method.isin([m.name for m in method_styles])], "frameworks", n_digits=n_digits)#, ["rank", "normalized_score", ])

    plot_figure(df, method_styles, figname="cdf-frameworks")

    plot_figure(
        df, [x for x in method_styles if "ensemble" not in x.name], figname="cdf-frameworks-tuned",
        title="Effect of tuning configurations",
    )

    plot_figure(
        df,
        [x for x in method_styles if any(pattern in x.name for pattern in ["tuned", "AutoGluon"])],
        figname="cdf-frameworks-ensemble",
        title="Effect of tuning & ensembling",
        # title="Comparison of frameworks",
    )

    cmap = matplotlib.colormaps["viridis"]
    # Plot effect number of training datasets
    method_styles = ag_styles + [
        MethodStyle(
            zeroshot_name(n_training_dataset=size),
            color=cmap(i / (len(n_training_datasets) - 1)), linestyle="-", label_str=r"$\mathcal{D}~=~" + f"{size}$",
        )
        for i, size in enumerate(n_training_datasets)
    ]
    plot_figure(df, method_styles, title="Effect of number of training tasks", figname="cdf-n-training-datasets")

    # # Plot effect number of training fold
    # method_styles = ag_styles + [
    #     MethodStyle(
    #         zeroshot_name(n_training_fold=size),
    #         color=cmap(i / (len(n_training_folds) - 1)), linestyle="-", label_str=f"S{size}",
    #     )
    #     for i, size in enumerate(n_training_folds)
    # ]
    # plot_figure(df, method_styles, title="Effect of number of training folds", figname="cdf-n-training-folds")

    # Plot effect number of portfolio size
    method_styles = ag_styles + [
        MethodStyle(
            zeroshot_name(n_portfolio=size),
            color=cmap(i / (len(n_portfolios) - 1)), linestyle="-", label_str=r"$\mathcal{N}~=~" + f"{size}$",
        )
        for i, size in enumerate(n_portfolios)
    ]
    plot_figure(df, method_styles, title="Effect of number of portfolio configurations", figname="cdf-n-configs")

    # Plot effect of number of training configurations
    method_styles = ag_styles + [
        MethodStyle(
            zeroshot_name(n_training_config=size),
            color=cmap(i / (len(n_training_configs) - 1)), linestyle="-", label_str=r"$\mathcal{M}'~=~" + f"{size}$",
        )
        for i, size in enumerate(n_training_configs)
    ]
    plot_figure(df, method_styles, title="Effect of number of offline configurations", figname="cdf-n-training-configs")

    # Plot effect of training time limit
    method_styles = ag_styles + [
        MethodStyle(
            zeroshot_name(max_runtime=size),
            color=cmap(i / (len(max_runtimes) - 1)), linestyle="-",
            label_str=time_suffix(size).replace("(", "").replace(")", "").strip(),
        )
        for i, size in enumerate(max_runtimes)
    ]
    plot_figure(df, method_styles, title="Effect of training time limit", figname="cdf-max-runtime")

    automl_frameworks = ["Autosklearn2", "Flaml", "Lightautoml", "H2oautoml"]
    for budget in ["1h", "4h"]:
        budget_suffix = f"\({budget}\)"
        # df = df[~df.method.str.contains("All")]
        df_selected = df[
            (df.method.str.contains(f"AutoGluon .*{budget_suffix}")) |
            (df.method.str.contains(".*(" + "|".join(automl_frameworks) + f").*{budget_suffix}")) |
            (df.method.str.contains(f"Portfolio-N{n_portfolios_default} .*{budget_suffix}")) |
            (df.method.str.contains(".*(" + "|".join(framework_types) + ")" + f".*{budget_suffix}")) |
            (df.method.str.contains(".*default.*"))
        ].copy()
        df_selected.method = df_selected.method.str.replace(f" {budget_suffix}", "").str.replace(f"\-N{n_portfolios_default}", "")
        show_latex_table(
            df_selected,
            f"selected-methods-{budget}",
            show_table=True,
            n_digits=n_digits,
        )

    show_latex_table(df[(df.method.str.contains("Portfolio") | (df.method.str.contains("AutoGluon ")))], "zeroshot", n_digits=n_digits)

    fig, _, bbox_extra_artists = show_scatter_performance_vs_time(df, metric_cols=["normalized-error", "rank"])
    fig_save_path = figure_path() / f"scatter-perf-vs-time.pdf"
    fig.savefig(fig_save_path, bbox_extra_artists=bbox_extra_artists, bbox_inches='tight')
    fig.show()

    fig, _, bbox_extra_artists = show_scatter_performance_vs_time_lower_budgets(df, metric_cols=["normalized-error", "rank"])
    fig_save_path = figure_path() / f"scatter-perf-vs-time-lower-budget.pdf"
    fig.savefig(fig_save_path, bbox_extra_artists=bbox_extra_artists, bbox_inches='tight')
    fig.show()

    plot_critical_diagrams(df)