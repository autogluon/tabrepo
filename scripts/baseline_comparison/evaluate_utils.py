from ast import literal_eval
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
    fig, axes = plt.subplots(len(metrics), len(dimensions), sharex='col', sharey='row', figsize=(9, 2), dpi=300)

    for i, (dimension, legend) in enumerate(dimensions):

        for j, metric in enumerate(metrics):
            df_portfolio = df.loc[df.method.str.contains(f"Portfolio-N.*-{dimension}.*4h"), :].copy()
            df_portfolio["is_ensemble"] = df_portfolio.method.str.contains("(ensemble)")
            df_ag = df.loc[df.method.str.contains("AutoGluon best \(4h\)"), metric].copy()
            df_askl2 = df.loc[df.method.str.contains("Autosklearn2 \(4h\)"), metric].copy()

            df_portfolio.loc[df_portfolio["is_ensemble"], dimension] = df_portfolio.loc[df_portfolio["is_ensemble"], "method"].apply(
                lambda s: int(s.replace(" (ensemble) (4h)", "").split("-")[-1][1:]))
            df_portfolio.loc[~df_portfolio["is_ensemble"], dimension] = df_portfolio.loc[~df_portfolio["is_ensemble"], "method"].apply(
                lambda s: int(s.replace(" (4h)", "").split("-")[-1][1:]))

            # Drop instances where dimension=1
            df_portfolio = df_portfolio.loc[df_portfolio[dimension] != 1, :]

            if len(metrics) > 1:
                ax = axes[j][i]
            else:
                ax = axes[i]

            for is_ens in [False, True]:
                df_portfolio_agg = df_portfolio.loc[df_portfolio["is_ensemble"] == is_ens].copy()
                df_portfolio_agg = df_portfolio_agg[[dimension, metric, "seed"]].groupby([dimension, "seed"]).mean()[metric]
                dim, mean, sem = df_portfolio_agg.groupby(dimension).agg(["mean", "sem"]).reset_index().values.T
                # _, _, std = df_portfolio_agg.groupby(dimension).agg(["mean", "std"]).reset_index().values.T

                label = "Portfolio"
                if is_ens:
                    label += " (ensemble)"

                ax.plot(
                    dim, mean,
                    label=label,
                    linestyle="-",
                    marker="o",
                    linewidth=0.7,
                )

                ax.fill_between(
                    dim,
                    [m - s for m, s in zip(mean, sem)],
                    [m + s for m, s in zip(mean, sem)],
                    alpha=0.2,
                )

            ax.set_xlim([0, 200])
            if j == len(metrics) - 1:
                ax.set_xlabel(legend)
            if i == 0:
                ax.set_ylabel(f"{metric}")
            ax.grid()
            ax.hlines(df_ag.mean(), xmin=0, xmax=max(dim), color="black", label="AutoGluon", ls="--")
            ax.hlines(df_askl2.mean(), xmin=0, xmax=max(dim), color="darkgray", label="Autosklearn2", ls="--")
            if i == 1 and j == 0:
                ax.legend()
                # specify order
                order = [0, 1, 3, 2]

                # reordering the labels
                handles, labels = ax.get_legend_handles_labels()

                # pass handle & labels lists along with order as below
                ax.legend([handles[i] for i in order], [labels[i] for i in order])


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
    df: pd.DataFrame,
    framework_types: list,
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


def get_framework_type_method_names(framework_types, max_runtime=4 * 3600):
    f_map = dict()
    f_map_type = dict()
    f_map_inverse = dict()
    for framework_type in framework_types:
        f_tuned = framework_name(framework_type, max_runtime=max_runtime, ensemble_size=1, tuned=True)
        f_tuned_ensembled = framework_name(framework_type, max_runtime=max_runtime, tuned=True)
        f_default = framework_name(framework_type, tuned=False)
        f_map[framework_type] = dict(
            default=f_default,
            tuned=f_tuned,
            tuned_ensembled=f_tuned_ensembled,
        )
        f_map_inverse[f_default] = "default"
        f_map_inverse[f_tuned] = "tuned"
        f_map_inverse[f_tuned_ensembled] = "tuned_ensembled"
        f_map_type[f_default] = framework_type
        f_map_type[f_tuned] = framework_type
        f_map_type[f_tuned_ensembled] = framework_type
    return f_map, f_map_type, f_map_inverse


# FIXME: Avoid hardcoding
def plot_tuning_impact(df: pd.DataFrame, framework_types: list, save_prefix: str, show: bool = True):
    df = df.copy()

    f_map, f_map_type, f_map_inverse = get_framework_type_method_names(framework_types=framework_types)

    metric = "normalized-error"

    df["framework_type"] = df["method"].map(f_map_type).fillna(df["method"])
    df["tune_method"] = df["method"].map(f_map_inverse).fillna("default")

    framework_types = ["AutoGluon best (4h)", "Autosklearn2 (4h)"] + framework_types

    df_plot = df[df["framework_type"].isin(framework_types)]

    df_plot_w_mean = df_plot[["framework_type", "tune_method", metric, "dataset", "fold"]].groupby(
        ["framework_type", "tune_method", "dataset"]
    )[metric].mean().reset_index()
    df_plot_w_mean_2 = df_plot.groupby(["framework_type", "tune_method"])[metric].mean().reset_index()
    df_plot_w_mean_2 = df_plot_w_mean_2.sort_values(by=metric, ascending=False)
    df_plot_w_mean_2 = df_plot_w_mean_2[df_plot_w_mean_2["tune_method"] == "default"]
    baseline_mean = df_plot_w_mean_2[df_plot_w_mean_2["framework_type"] == "AutoGluon best (4h)"][metric].iloc[0]
    df_plot_w_mean_2 = df_plot_w_mean_2[df_plot_w_mean_2["framework_type"] != "AutoGluon best (4h)"]
    askl2_mean = df_plot_w_mean_2[df_plot_w_mean_2["framework_type"] == "Autosklearn2 (4h)"][metric].iloc[0]
    df_plot_w_mean_2 = df_plot_w_mean_2[df_plot_w_mean_2["framework_type"] != "Autosklearn2 (4h)"]
    framework_type_order = list(df_plot_w_mean_2["framework_type"].values)

    # boxplot
    # sns.set_color_codes("pastel")
    with sns.axes_style("whitegrid"):
        colors = sns.color_palette("pastel").as_hex()
        errcolors = sns.color_palette("deep").as_hex()
        fig, ax = plt.subplots(1, 1, figsize=(12, 4.5))
        sns.barplot(
            x="framework_type", y=metric,
            # hue="tune_method",  # palette=["m", "g", "r],
            label="Default",
            data=df_plot[df_plot["tune_method"] == "default"], ax=ax,
            order=framework_type_order, color=colors[0],
            width=0.8,
            errcolor=errcolors[0],
        )
        sns.barplot(
            x="framework_type", y=metric,
            # hue="tune_method",  # palette=["m", "g", "r],
            label="Tuned (4h)",
            data=df_plot[df_plot["tune_method"] == "tuned"], ax=ax,
            order=framework_type_order, color=colors[1],
            width=0.7,
            errcolor=errcolors[1],
        )
        boxplot = sns.barplot(
            x="framework_type", y=metric,
            # hue="tune_method",  # palette=["m", "g", "r],
            label="Tuned + Ensembled (4h)",
            data=df_plot[df_plot["tune_method"] == "tuned_ensembled"], ax=ax,
            order=framework_type_order, color=colors[2],
            width=0.6,
            errcolor=errcolors[2],
        )
        boxplot.set(xlabel=None)  # remove "Method" in the x-axis
        boxplot.set_title("Effect of tuning and ensembling")
        ax.axhline(y=baseline_mean, label="AutoGluon (4h)", color="black", linewidth=2.0, ls="--")
        ax.axhline(y=askl2_mean, label="Autosklearn2 (4h)", color="darkgray", linewidth=2.0, ls="--")
        ax.set_ylim([0, 1])

        # specify order
        order = [2, 3, 4, 1, 0]

        # reordering the labels
        handles, labels = plt.gca().get_legend_handles_labels()

        # pass handle & labels lists along with order as below
        plt.legend([handles[i] for i in order], [labels[i] for i in order])
        plt.tight_layout()

        if save_prefix:
            fig_path = figure_path(prefix=save_prefix)
            fig_save_path = fig_path / f"tuning-impact.pdf"
            plt.savefig(fig_save_path)
        if show:
            plt.show()


def plot_family_proportion(df, method="Portfolio-N200 (ensemble) (4h)", save_prefix: str = None, show: bool = True, hue_order: list = None):
    df_family = df[df["method"] == method].copy()
    df_family = df_family[df_family["fold"] == 0]
    portfolios = list(df_family["config_selected"].values)
    portfolios_lst = [literal_eval(portfolio) for portfolio in portfolios]

    from collections import defaultdict
    type_count = defaultdict(int)
    type_count_family = defaultdict(int)
    type_count_per_iter = dict()
    type_count_family_per_iter = dict()

    n_iters = 50
    for i in range(n_iters):
        type_count_per_iter[i] = defaultdict(int)
        type_count_family_per_iter[i] = defaultdict(int)
        for portfolio in portfolios_lst:
            if len(portfolio) <= i:
                continue
            name = portfolio[i]
            family = name.split('_', 1)[0]
            # To keep naming consistency in the paper
            if family == "NeuralNetTorch":
                family = "MLP"
            type_count[name] += 1
            type_count_family[family] += 1
            type_count_per_iter[i][name] += 1
            type_count_family_per_iter[i][family] += 1

    families = sorted(list(type_count_family.keys()))

    import copy
    type_count_cumulative = dict()
    type_count_family_cumulative = dict()
    type_count_cumulative[0] = copy.deepcopy(type_count_per_iter[0])
    type_count_family_cumulative[0] = copy.deepcopy(type_count_family_per_iter[0])
    for i in range(1, n_iters):
        type_count_cumulative[i] = copy.deepcopy(type_count_cumulative[i-1])
        for k in type_count_per_iter[i].keys():
            type_count_cumulative[i][k] += type_count_per_iter[i][k]
        type_count_family_cumulative[i] = copy.deepcopy(type_count_family_cumulative[i-1])
        for k in type_count_family_per_iter[i].keys():
            type_count_family_cumulative[i][k] += type_count_family_per_iter[i][k]

    data = []
    for i in range(n_iters):
        data.append([type_count_family_per_iter[i][f] for f in families])
    data_cumulative = []
    for i in range(n_iters):
        data_cumulative.append([type_count_family_cumulative[i][f] for f in families])


    data_df = pd.DataFrame(data=data, columns=families)
    data_df = data_df.div(data_df.sum(axis=1), axis=0) * 100
    data_df2 = data_df.stack().reset_index(name='Model Frequency (%) at Position').rename(columns={'level_1': 'Model', 'level_0': 'Portfolio Position'})
    data_df2["Portfolio Position"] += 1

    data_cumulative_df = pd.DataFrame(data=data_cumulative, columns=families)
    data_cumulative_df = data_cumulative_df.div(data_cumulative_df.sum(axis=1), axis=0) * 100
    data_cumulative_df2 = data_cumulative_df.stack().reset_index(name='Cumulative Model Frequency (%)').rename(columns={'level_1': 'Model', 'level_0': 'Portfolio Position'})
    data_cumulative_df2["Portfolio Position"] += 1
    fig, axes = plt.subplots(2, 1, sharey=False, sharex=True, figsize=(16, 10), dpi=300, layout="constrained")

    sns.histplot(
        data_df2,
        x="Portfolio Position",
        weights="Model Frequency (%) at Position",
        # stat="percent",
        hue="Model",
        hue_order=hue_order,
        multiple="stack",
        # palette="light:m_r",
        palette="pastel",
        edgecolor=".3",
        linewidth=.5,
        discrete=True,
        ax=axes[0],
        # legend=False,
    )
    axes[0].set(ylabel="Model Frequency (%) at Position")
    axes[0].set_xlim([0, n_iters+1])
    axes[0].set_ylim([0, 100])
    sns.move_legend(axes[0], "upper left")

    sns.histplot(
        data_cumulative_df2,
        x="Portfolio Position",
        weights="Cumulative Model Frequency (%)",
        # stat="percent",
        hue="Model",
        hue_order=hue_order,
        multiple="stack",
        # palette="light:m_r",
        palette="pastel",
        edgecolor=".3",
        linewidth=.5,
        discrete=True,
        ax=axes[1],
        legend=False,
    )
    axes[1].set(ylabel="Cumulative Model Frequency (%)")
    axes[1].set_xlim([0, n_iters+1])
    axes[1].set_ylim([0, 100])

    fig.suptitle(f"Model Family Presence in Portfolio by Training Order")

    if save_prefix:
        fig_path = figure_path(prefix=save_prefix)
        fig_save_path = fig_path / f"portfolio-model-presence.pdf"
        plt.savefig(fig_save_path)
    if show:
        plt.show()
