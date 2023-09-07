from dataclasses import dataclass
from scripts import output_path

import matplotlib.pyplot as plt
from typing import List
import numpy as np
import pandas as pd

from scripts.baseline_comparison.baselines import zeroshot_name, framework_types


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
    s = df_metrics.to_latex(float_format="%.2f")
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
        avg_metric = df.groupby("method").agg("mean")[metric]
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
                if i == 0:
                    axes[i].set_ylabel(f"CDF")
            else:
                print(f"Could not find method {method_style.name}")
    axes[-1].legend(fontsize="small")
    return fig, axes


def show_scatter_performance_vs_time(df: pd.DataFrame, max_runtimes, metric_col):
    import seaborn as sns

    df_metrics = compute_avg_metrics(df)
    colors = [sns.color_palette("bright")[j] for j in range(10)]
    colors[1] = "black"
    markers = ["*", 's', 'v', '^', "8", "D"]
    # cash_methods = df_metrics.index.str.match("All \(.* samples.*ensemble\)")
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(10, 3))

    df_frameworks = {
        # gets methods such as Zeroshot-N20 (1.0h), would be cleaner to use a regexp
        "Zeroshot": df_metrics[df_metrics.index.str.contains("Zeroshot.*\(.*h\)")],
        "AutoGluon": df_metrics[df_metrics.index.str.contains("AutoGluon ")]
    }
    automl_frameworks = ["Autosklearn2", "Flaml", "Lightautoml", "H2oautoml"]
    for automl_framework in automl_frameworks:
        df_frameworks[automl_framework] = df_metrics[df_metrics.index.str.contains(automl_framework)]

    for i, metric in enumerate(
            [
                "time-train-s",
                "time-infer-s",
            ]
    ):
        for j, (framework, df_framework) in enumerate(df_frameworks.items()):
            axes[i].scatter(
                df_framework[metric],
                df_framework[metric_col],
                label=framework,
                color=colors[j],
                marker=markers[j],
                s=50.0 if markers[j] == "*" else None,
            )
            axes[i].set_xlabel(metric)
            if i == 0:
                axes[i].set_ylabel(metric_col)
    # fig.legend(axes, df_frameworks.keys(), loc = "upper center", ncol=5)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, df_frameworks.keys(), loc='upper center', ncol=len(df_frameworks),)
    fig.tight_layout()
    # fig.legend(handles, df_frameworks.keys(), loc='right center', ncol=1)
    # plt.legend(loc="upper center")
    return fig, axes
