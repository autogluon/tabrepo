from __future__ import annotations
from autorank import autorank, plot_stats, create_report, latex_table

from dataclasses import dataclass
from scripts import output_path

import matplotlib.pyplot as plt
from typing import List
import numpy as np
import pandas as pd

from scripts.baseline_comparison.baselines import n_portfolios_default


@dataclass
class MethodStyle:
    name: str
    color: str
    linestyle: str = None  # linestyle of the method, default to plain
    linewidth: float = None
    label: bool = True  # whether to show the method name as label
    label_str: str = None


def iqm(x):
    x = list(sorted(x))
    start = len(x) * 1 // 4
    end = len(x) * 3 // 4
    return np.mean(x[start:end])


def show_latex_table(df: pd.DataFrame, title: str, show_table: bool = False, latex_kwargs=None, n_digits=None, save_prefix=None):
    metrics = ["normalized-error", "rank", "time fit (s)", "time infer (s)"]
    df_metrics = compute_avg_metrics(df, metrics)
    save_latex_table(df=df_metrics, title=title, show_table=show_table, latex_kwargs=latex_kwargs, n_digits=n_digits, save_prefix=save_prefix)


def figure_path(prefix: str = None, suffix: str = None):
    fig_save_path_dir = output_path
    if prefix:
        fig_save_path_dir = fig_save_path_dir / prefix
    fig_save_path_dir = fig_save_path_dir / "figures"
    if suffix:
        fig_save_path_dir = fig_save_path_dir / suffix
    fig_save_path_dir.mkdir(parents=True, exist_ok=True)
    return fig_save_path_dir


def table_path(prefix: str = None, suffix: str = None):
    table_save_path_dir = output_path
    if prefix:
        table_save_path_dir = table_save_path_dir / prefix
    table_save_path_dir = table_save_path_dir / "tables"
    if suffix:
        table_save_path_dir = table_save_path_dir / suffix
    table_save_path_dir.mkdir(parents=True, exist_ok=True)
    return table_save_path_dir


def save_latex_table(df: pd.DataFrame, title: str, show_table: bool = False, latex_kwargs: dict | None = None, n_digits = None, save_prefix: str = None):
    if n_digits:
        for col in df.columns:
            if (not df[col].dtype == "object") and (not df[col].dtype == "int64"):
                n_digit = n_digits.get(col, 2)
                df.loc[:, col] = df.loc[:, col].apply(lambda s: f'{s:.{n_digit}f}')

    if latex_kwargs is None:
        latex_kwargs = dict()
    s = df.to_latex(**latex_kwargs)
    latex_folder = table_path(prefix=save_prefix)
    latex_file = latex_folder / f"{title}.tex"
    print(f"Writing latex result in {latex_file}")
    with open(latex_file, "w") as f:
        f.write(s)
    if show_table:
        print(s)


def compute_avg_metrics(df: pd.DataFrame, metrics):
    avg_metrics = {}
    for metric in metrics:
        avg_metric = df.loc[:, ["method", metric]].groupby("method").agg("mean")[metric]
        #
        # # We use mean to aggregate runtimes as IQM does not make too much sense in this context,
        # # it makes only sense to aggregate y-metrics such as normalized scores or ranks.
        # if "time" in metric:
        #     avg_metric = df.groupby("method").agg("mean")[metric]
        # else:
        #     avg_metric = df.groupby("method").agg(iqm)[metric]
        avg_metric.sort_values().head(60)
        xx = avg_metric.sort_values()
        avg_metrics[metric] = xx

    # avg_metric = df.groupby("method").agg("max")["time_train_s"]
    # avg_metric.sort_values().head(60)
    # avg_metrics["time_train_s (max)"] = avg_metric.sort_values()

    df_metrics = pd.DataFrame(avg_metrics).sort_values(by="normalized-error")
    df_metrics.columns = [x.replace("_", "-") for x in df_metrics.columns]
    return df_metrics


def show_cdf(df: pd.DataFrame, method_styles: List[MethodStyle] = None):
    if method_styles is None:
        method_styles = [
            MethodStyle(method, color=None, linestyle=None, label=method)
            for method in df.method.unique()
        ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    metrics = ["normalized-error", "rank"]
    for i, metric in enumerate(metrics):
        for j, method_style in enumerate(method_styles):
            xx = df.loc[df.method == method_style.name, metric].sort_values()
            if len(xx) > 0:
                if method_style.label:
                    label = (
                        method_style.label_str
                        if method_style.label_str
                        else method_style.name
                    )
                else:
                    label = None
                axes[i].plot(
                    xx.values,
                    np.arange(len(xx)) / len(xx),
                    # label=method_style.name if method_style.label else None,
                    label=label,
                    color=method_style.color,
                    linestyle=method_style.linestyle,
                    lw=method_style.linewidth if method_style.linestyle else 1.5,
                )
                # axes[i].set_title(metric.replace("_", "-"))
                axes[i].set_xlabel(metric.replace("_", "-"))
                axes[i].grid('on')
                if i == 0:
                    axes[i].set_ylabel(f"CDF")
            else:
                print(f"Could not find method {method_style.name}")
    axes[-1].legend(fontsize="small")
    return fig, axes


def show_scatter_performance_vs_time(df: pd.DataFrame, metric_cols):
    # show performance over time for 1h, 4h and 24h budgets for AG, PF and other frameworks
    import seaborn as sns
    plt.rcParams.update({'font.size': 16})

    df_metrics = compute_avg_metrics(df, ["normalized-error", "rank", "time fit (s)", "time infer (s)", "fit budget"])
    colors = [sns.color_palette("pastel")[j] for j in range(10)]

    # makes autogluon black to respect colors used in previous plots
    colors[6] = "black"
    colors[7] = sns.color_palette("bright")[6]
    # colors[6] = "yellow"
    markers = ['x', 'v', '^', "8", "D", 'v', "s", '*', ]
    # cash_methods = df_metrics.index.str.match("All \(.* samples.*ensemble\)")
    fig, axes = plt.subplots(1, 2, sharey=False, sharex=True, figsize=(14, 3), dpi=300)

    df_frameworks = {}
    df_frameworks["CatBoost (tuned + ens)"] = df_metrics[df_metrics.index.str.startswith("CatBoost (tuned + ensemble)")]
    df_frameworks["Autosklearn"] = df_metrics[df_metrics.index.str.startswith("Autosklearn ")]
    for automl_framework in ["Autosklearn2", "Flaml", "Lightautoml", "H2oautoml"]:
        df_frameworks[automl_framework] = df_metrics[df_metrics.index.str.contains(automl_framework)]

    df_frameworks["AutoGluon best"] = df_metrics[df_metrics.index.str.contains("AutoGluon best.*h")]
    df_frameworks["Portfolio"] = df_metrics[df_metrics.index.str.contains( f"Portfolio-N{n_portfolios_default} .*ens.*h")]

    for i, metric_col in enumerate(metric_cols):
        for j, (framework, df_framework) in enumerate(df_frameworks.items()):
            df_framework_sorted = df_framework.sort_values(by="fit budget", ascending=True)
            fitting_budget_hour = df_framework_sorted["fit budget"] / 3600

            axes[i].scatter(
                fitting_budget_hour,
                df_framework_sorted[metric_col],
                label=framework,
                color=colors[j],
                marker=markers[j],
                s=100 if markers[j] == "*" else 70,
            )
            axes[i].set_xlabel("Fitting budget (time)")
            axes[i].set_ylabel(metric_col)
            axes[i].set_xscale("log")
            # axes[i].set_xticks([5/60, 10/60, 30/60, 1, 4, 24], ["5m", "10m", "30m", "1h", "4h", "24h"])
            axes[i].set_xticks([1, 4, 24], ["1h", "4h", "24h"])
            axes[i].grid('on')

    # fig.legend(axes, df_frameworks.keys(), loc = "upper center", ncol=5)
    handles, labels = axes[-1].get_legend_handles_labels()
    # fig.legend(handles, df_frameworks.keys(), loc='center right'), #ncol=len(df_frameworks),)

    # used https://stackoverflow.com/questions/25068384/bbox-to-anchor-and-loc-in-matplotlib
    # to be able to save figure with legend outside of bbox
    lgd = fig.legend(handles, df_frameworks.keys(), loc='upper center', ncol=4,
                     bbox_to_anchor=(0.5, 1.2))
    text = fig.text(-0.2, 1.05, "", transform=axes[0].transAxes)
    bbox_extra_artists = (lgd, text)

    return fig, axes, bbox_extra_artists


def show_scatter_performance_vs_time_lower_budgets(df: pd.DataFrame, metric_cols):
    # show performance over time for all budgets for AG and PF where evaluations are available
    import seaborn as sns
    from scripts.baseline_comparison.plot_utils import compute_avg_metrics
    n_portfolios_default = 200
    metric_cols = ["normalized-error", "rank"]
    df_metrics = compute_avg_metrics(df, ["normalized-error", "rank", "time fit (s)", "time infer (s)", "fit budget"])

    # makes autogluon black to respect colors used in previous plots
    colors = ["black", sns.color_palette("bright")[6]]
    # colors[6] = "yellow"
    markers = ["s", '*', ]
    # cash_methods = df_metrics.index.str.match("All \(.* samples.*ensemble\)")
    fig, axes = plt.subplots(1, 2, sharey=False, sharex=True, figsize=(10, 3), dpi=300)

    df_frameworks = {}
    df_frameworks["AutoGluon best"] = df_metrics[df_metrics.index.str.contains("AutoGluon best")]
    df_frameworks["Portfolio"] = df_metrics[df_metrics.index.str.contains(f"Portfolio-N{n_portfolios_default} .*ens")]

    for i, metric_col in enumerate(metric_cols):
        for j, (framework, df_framework) in enumerate(df_frameworks.items()):
            df_framework_sorted = df_framework.sort_values(by="fit budget", ascending=True)
            fitting_budget_hour = df_framework_sorted["fit budget"] / 3600

            axes[i].scatter(
                fitting_budget_hour,
                df_framework_sorted[metric_col],
                label=framework,
                color=colors[j],
                marker=markers[j],
                s=100 if markers[j] == "*" else 70,
            )
            axes[i].set_xlabel("Fitting budget (time)")
            axes[i].set_ylabel(metric_col)
            axes[i].set_xscale("log")
            axes[i].set_xticks([5 / 60, 10 / 60, 30 / 60, 1, 4, 24], ["5m", "10m", "30m", "1h", "4h", "24h"])
            axes[i].grid('on')
            if i == 1:
                axes[i].legend()

    # fig.legend(axes, df_frameworks.keys(), loc = "upper center", ncol=5)
    handles, labels = axes[-1].get_legend_handles_labels()
    # fig.legend(handles, df_frameworks.keys(), loc='center right'), #ncol=len(df_frameworks),)

    # used https://stackoverflow.com/questions/25068384/bbox-to-anchor-and-loc-in-matplotlib
    # to be able to save figure with legend outside of bbox
    lgd = fig.legend(handles, df_frameworks.keys(), loc='upper center', ncol=len(df_frameworks),
                     bbox_to_anchor=(0.5, 1.1))
    text = fig.text(-0.2, 1.05, "", transform=axes[0].transAxes)
    bbox_extra_artists = (lgd, text)

    return fig, axes, bbox_extra_artists


def plot_critical_diagrams(df, save_prefix: str = None):
    plt.rcParams.update({'font.size': 12})

    fig, axes = plt.subplots(1, 2, figsize=(12, 2))
    for i, budget in enumerate(["1h", "4h"]):  # , "24h"]):
        df_sub = df[df.method.str.contains(budget)].copy()
        df_sub.loc[:, "method"] = df_sub.loc[:, "method"].apply(
            lambda s: s.replace("-N200", "").replace(f" ({budget})", ""))
        df_sub.loc[:, "method"] = df_sub.loc[:, "method"].apply(
            lambda s: s.replace("ensemble", "ens"))

        df_sub = df_sub[df_sub.method.isin([
            f'Portfolio (ens)',
            f'AutoGluon best',
            f'H2oautoml',
            f'Autosklearn2',
            f'Autosklearn',
            f'Flaml',
            f'Lightautoml',
            f'CatBoost (tuned + ens)',
        ])]
        data = df_sub.pivot_table(index="dataset", columns="method", values="rank")
        result = autorank(data, alpha=0.05, verbose=False, order="ascending", force_mode="nonparametric")
        ax = axes[i]
        ax.set_title(budget)
        plot_stats(result, ax=ax, width=4, allow_insignificant=True)

    fig_path = figure_path(prefix=save_prefix)
    fig_save_path = fig_path / f"critical-diagram.pdf"
    plt.tight_layout()
    plt.savefig(fig_save_path)
    plt.show()
