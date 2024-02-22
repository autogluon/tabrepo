from typing import List, Callable, Dict
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from dataclasses import dataclass

from tabrepo import EvaluationRepository
from tabrepo.utils.cache import cache_function_dataframe
from tabrepo.utils.normalized_scorer import NormalizedScorer
from tabrepo.utils.rank_utils import RankScorer

from scripts import output_path
from scripts.baseline_comparison.baselines import ResultRow, zeroshot_name, time_suffix, framework_name
from scripts.baseline_comparison.plot_utils import (
    MethodStyle,
    show_cdf,
    figure_path,
    table_path,
)


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


def make_scorers(repo: EvaluationRepository, only_baselines=False):
    if only_baselines:
        df_results_baselines = repo._zeroshot_context.df_baselines
    else:
        df_results_baselines = pd.concat([
            repo._zeroshot_context.df_configs_ranked,
            repo._zeroshot_context.df_baselines,
        ], ignore_index=True)

    unique_dataset_folds = [
        f"{repo.dataset_to_tid(dataset)}_{fold}"
        for dataset in repo.datasets()
        for fold in range(repo.n_folds())
    ]
    rank_scorer = RankScorer(df_results_baselines, tasks=unique_dataset_folds, pct=False)
    normalized_scorer = NormalizedScorer(df_results_baselines, tasks=unique_dataset_folds, baseline=None)
    return rank_scorer, normalized_scorer


def impute_missing(repo: EvaluationRepository):
    # impute random forest data missing folds by picking data from another fold
    # TODO remove once we have complete dataset
    df = repo._zeroshot_context.df_configs_ranked
    df["framework_type"] = df.apply(lambda row: row["framework"].split("_")[0], axis=1)

    missing_tasks = [(3583, 0), (58, 9), (3483, 0)]
    for tid, fold in missing_tasks:
        impute_fold = (fold + 1) % 10
        df_impute = df[(df.framework_type == 'RandomForest') & (df.dataset == f"{tid}_{impute_fold}")].copy()
        df_impute['dataset'] = f"{tid}_{fold}"
        df_impute['fold'] = fold
        df = pd.concat([df, df_impute], ignore_index=True)
    repo._zeroshot_context.df_configs_ranked = df


def plot_figure(df, method_styles: List[MethodStyle], title: str = None, figname: str = None, show: bool = False, save_prefix: str = None, format: str = "pdf"):
    fig, _ = show_cdf(df[df.method.isin([m.name for m in method_styles])], method_styles=method_styles)
    if title:
        fig.suptitle(title)
    if figname:
        fig_path = figure_path(prefix=save_prefix)
        fig_save_path = fig_path / f"{figname}.{format}"
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


def time_cutoff_baseline(df: pd.DataFrame, rank_scorer, normalized_scorer, repo, rel_tol: float = 0.1) -> pd.DataFrame:
    df = df.copy()
    # TODO Portfolio excess are due to just using one fold to simulate runtimes, fix it
    mask = (df["time fit (s)"] > df["fit budget"] * (1 + rel_tol)) & (~df.method.str.contains("Portfolio"))

    # gets performance of Extra-trees baseline on all tasks
    dd = repo._zeroshot_context.df_configs_ranked
    dd = dd[dd.framework == "ExtraTrees_c1_BAG_L1"]
    dd["tid"] = dd.dataset.apply(lambda s: int(s.split("_")[0]))
    dd["fold"] = dd.dataset.apply(lambda s: int(s.split("_")[1]))
    dd["rank"] = dd.apply(lambda row: rank_scorer.rank(task=row["dataset"], error=row["metric_error"]), axis=1)
    dd["normalized-score"] = dd.apply(
        lambda row: normalized_scorer.rank(dataset=row["dataset"], error=row["metric_error"]), axis=1)
    df_baseline = dd[["tid", "fold", "rank", "normalized-score"]]

    df.loc[mask, ["normalized_score", "rank"]] = df.loc[mask, ["tid", "fold"]].merge(df_baseline, on=["tid", "fold"])[
        ["normalized-score", "rank"]].values

    return df


def rename_dataframe(df):
    rename_dict = make_rename_dict(suffix="8c_2023_11_14")
    df["method"] = df["method"].replace(rename_dict)
    df.rename({
        "normalized_error": "normalized-error",
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
    df.method = df.method.str.replace("NeuralNetTorch", "MLP")
    return df


def generate_sensitivity_plots(df, show: bool = False, save_prefix: str = None):
    dimensions = [
        ("M", "Number of configurations per family"),
        ("D", "Number of training datasets to fit portfolios"),
    ]
    metrics = [
        "normalized-error",
        # "rank",
    ]

    # show stds
    fig, axes = plt.subplots(len(metrics), len(dimensions), sharex='col', sharey='row', figsize=(9, 4))

    for i, (dimension, legend) in enumerate(dimensions):

        for j, metric in enumerate(metrics):
            df_portfolio = df.loc[df.method.str.contains(f"Portfolio-N.*-{dimension}.*4h"), :].copy()
            df_portfolio["is_ensemble"] = df_portfolio.method.str.contains("(ensemble)")
            df_ag = df.loc[df.method.str.contains("AutoGluon best \(4h\)"), metric].copy()

            df_portfolio.loc[df_portfolio["is_ensemble"], dimension] = df_portfolio.loc[df_portfolio["is_ensemble"], "method"].apply(
                lambda s: int(s.replace(" (ensemble) (4h)", "").split("-")[-1][1:]))
            df_portfolio.loc[~df_portfolio["is_ensemble"], dimension] = df_portfolio.loc[~df_portfolio["is_ensemble"], "method"].apply(
                lambda s: int(s.replace(" (4h)", "").split("-")[-1][1:]))

            if len(metrics) > 1:
                ax = axes[j][i]
            else:
                ax = axes[i]

            for is_ens in [True, False]:
                df_portfolio_agg = df_portfolio.loc[df_portfolio["is_ensemble"] == is_ens].copy()
                df_portfolio_agg = df_portfolio_agg[[dimension, metric, "seed"]].groupby([dimension, "seed"]).mean()[metric]
                dim, mean, sem = df_portfolio_agg.groupby(dimension).agg(["mean", "sem"]).reset_index().values.T

                label = "Portfolio"
                if is_ens:
                    label += " (ensemble)"

                ax.errorbar(
                    dim, mean, sem,
                    label=label,
                    linestyle="",
                    marker="o",
                )
            ax.set_xlim([0, None])
            if j == len(metrics) - 1:
                ax.set_xlabel(legend)
            if i == 0:
                ax.set_ylabel(f"{metric}")
            ax.grid()
            ax.hlines(df_ag.mean(), xmin=0, xmax=max(dim), color="black", label="AutoGluon", ls="--")
            if i == 1 and j == 0:
                ax.legend()
    fig_path = figure_path(prefix=save_prefix)
    fig_save_path = fig_path / f"sensitivity.pdf"
    plt.tight_layout()
    plt.savefig(fig_save_path)
    if show:
        plt.show()


def save_total_runtime_to_file(total_time_h, save_prefix: str = None):
    # Save total runtime so that "show_repository_stats.py" can show the ratio of saved time
    path = table_path(prefix=save_prefix)
    with open(path / "runtime.txt", "w") as f:
        f.write(str(total_time_h))


def plot_ctf(
    df,
    framework_types,
    expname_outdir,
    n_training_datasets,
    n_portfolios,
    n_training_configs,
    max_runtimes,
):
    linestyle_ensemble = "--"
    linestyle_default = "-"
    linestyle_tune = "dotted"
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
                color=sns.color_palette('bright', n_colors=20)[i],
                linestyle=linestyle_default,
                label=True,
                label_str=framework_type,
            )
        )
        method_styles.append(
            MethodStyle(
                framework_name(framework_type, max_runtime=4 * 3600, ensemble_size=1, tuned=True),
                color=sns.color_palette('bright', n_colors=20)[i],
                linestyle=linestyle_tune,
                label=False,
            )
        )
        method_styles.append(
            MethodStyle(
                framework_name(framework_type, max_runtime=4 * 3600, tuned=True),
                color=sns.color_palette('bright', n_colors=20)[i],
                linestyle=linestyle_ensemble,
                label=False,
                label_str=framework_type
            )
        )

    plot_figure(df, method_styles, figname="cdf-frameworks", save_prefix=expname_outdir)

    plot_figure(
        df, [x for x in method_styles if "ensemble" not in x.name], figname="cdf-frameworks-tuned",
        title="Effect of tuning configurations",
        save_prefix=expname_outdir,
    )

    plot_figure(
        df,
        [x for x in method_styles if any(pattern in x.name for pattern in ["tuned", "AutoGluon"])],
        figname="cdf-frameworks-ensemble",
        title="Effect of tuning & ensembling",
        # title="Comparison of frameworks",
        save_prefix=expname_outdir,
    )

    cmap = matplotlib.colormaps["viridis"]
    # Plot effect number of training datasets
    method_styles = ag_styles + [
        MethodStyle(
            zeroshot_name(n_training_dataset=size),
            color=cmap(i / (len(n_training_datasets) - 1)), linestyle="-", label_str=r"$\mathcal{D}'~=~" + f"{size}$",
        )
        for i, size in enumerate(n_training_datasets)
    ]
    plot_figure(df, method_styles, title="Effect of number of training tasks", figname="cdf-n-training-datasets", save_prefix=expname_outdir)

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
    plot_figure(df, method_styles, title="Effect of number of portfolio configurations", figname="cdf-n-configs", save_prefix=expname_outdir)

    # Plot effect of number of training configurations
    method_styles = ag_styles + [
        MethodStyle(
            zeroshot_name(n_training_config=size),
            color=cmap(i / (len(n_training_configs) - 1)), linestyle="-", label_str=r"$\mathcal{M}'~=~" + f"{size}$",
        )
        for i, size in enumerate(n_training_configs)
    ]
    plot_figure(df, method_styles, title="Effect of number of offline configurations", figname="cdf-n-training-configs", save_prefix=expname_outdir)

    # Plot effect of training time limit
    method_styles = ag_styles + [
        MethodStyle(
            zeroshot_name(max_runtime=size),
            color=cmap(i / (len(max_runtimes) - 1)), linestyle="-",
            label_str=time_suffix(size).replace("(", "").replace(")", "").strip(),
        )
        for i, size in enumerate(max_runtimes)
    ]
    plot_figure(df, method_styles, title="Effect of training time limit", figname="cdf-max-runtime", save_prefix=expname_outdir)

