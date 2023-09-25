from __future__ import annotations

from dataclasses import dataclass
from scripts import output_path

import matplotlib.pyplot as plt
from typing import List
import numpy as np
import pandas as pd

from scripts.baseline_comparison.baselines import zeroshot_name, framework_types, n_portfolios_default


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


def show_latex_table(df: pd.DataFrame, title: str, show_table: bool = False):
    df_metrics = compute_avg_metrics(df)
    latex_kwargs = dict(float_format="%.2f")
    save_latex_table(df=df_metrics, title=title, show_table=show_table, latex_kwargs=latex_kwargs)


def save_latex_table(df: pd.DataFrame, title: str, show_table: bool = False, latex_kwargs: dict | None = None):
    if latex_kwargs is None:
        latex_kwargs = dict()
    s = df.to_latex(**latex_kwargs)
    latex_folder = output_path / "tables"
    latex_folder.mkdir(exist_ok=True)
    latex_file = latex_folder / f"{title}.tex"
    print(f"Writing latex result in {latex_file}")
    with open(latex_file, "w") as f:
        f.write(s)
    if show_table:
        print(s)


def compute_avg_metrics(df: pd.DataFrame):
    avg_metrics = {}
    for metric in ["normalized_score", "rank", "time_train_s", "time_infer_s"]:
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

    df_metrics = pd.DataFrame(avg_metrics).sort_values(by="normalized_score")
    df_metrics.columns = [x.replace("_", "-") for x in df_metrics.columns]
    return df_metrics


def show_cdf(df: pd.DataFrame, method_styles: List[MethodStyle] = None):
    if method_styles is None:
        method_styles = [
            MethodStyle(method, color=None, linestyle=None, label=method)
            for method in df.method.unique()
        ]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
    metrics = ["normalized_score", "rank"]
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
    import seaborn as sns

    df_metrics = compute_avg_metrics(df)
    colors = [sns.color_palette("bright")[j] for j in range(10)]

    # makes autogluon black to respect colors used in previous plots
    colors[5] = "black"
    markers = ['v', '^', "8", "D", 'v', "s", '*', ]
    # cash_methods = df_metrics.index.str.match("All \(.* samples.*ensemble\)")
    fig, axes = plt.subplots(1, 2, sharey=False, sharex=True, figsize=(10, 3), dpi=300)

    df_frameworks = {}
    df_frameworks["Autosklearn"] = df_metrics[df_metrics.index.str.startswith("Autosklearn ")]
    for automl_framework in ["Autosklearn2", "Flaml", "Lightautoml", "H2oautoml"]:
        df_frameworks[automl_framework] = df_metrics[df_metrics.index.str.contains(automl_framework)]

    df_frameworks["AutoGluon best"] = df_metrics[df_metrics.index.str.contains("AutoGluon best")]
    df_frameworks["Portfolio"] = df_metrics[df_metrics.index.str.contains(f"Portfolio-N{n_portfolios_default} .*ensemble.*\(.*\)")]


    for i, metric_col in enumerate(metric_cols):
        for j, (framework, df_framework) in enumerate(df_frameworks.items()):
            # ugly way to get hour Portfolio-N160 + ensemble (0.17h) -> 0.17
            fitting_budget_hour = [float(s.split("(")[-1][:-2]) for s in df_framework.index]

            # Convert minutes to hours
            fitting_budget_hour = [f/60 if s.rsplit(")")[0][-1] == 'm' else f for s, f in zip(df_framework.index, fitting_budget_hour)]

            axes[i].scatter(
                fitting_budget_hour,
                df_framework[metric_col],
                label=framework,
                color=colors[j],
                marker=markers[j],
                s=100.0 if markers[j] == "*" else 70.0,
            )
            axes[i].set_xlabel("Fitting budget (time)")
            axes[i].set_ylabel(metric_col)
            axes[i].set_xscale("log")
            axes[i].set_xticks([5/60, 10/60, 30/60, 1, 4, 24], ["5m", "10m", "30m", "1h", "4h", "24h"])
            axes[i].grid('on')

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
