from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from typing import List
import numpy as np
import pandas as pd

@dataclass
class MethodStyle:
    name: str
    color: str
    linestyle: str = None  # linestyle of the method, default to plain
    label: bool = True  # whether to show the method name as label

def show_latex_table(df: pd.DataFrame):
    avg_metrics = {}
    for metric in ["rank", "normalized_score"]:
        avg_metric = df.groupby("method").mean(numeric_only=True)[metric]
        avg_metric.sort_values().head(60)
        xx = avg_metric.sort_values()
        avg_metrics[metric] = xx
    print(pd.DataFrame(avg_metrics).sort_values(by="rank").to_latex(float_format="%.2f"))

def show_cdf(df: pd.DataFrame, method_styles: List[MethodStyle] = None):
    if method_styles is None:
        method_styles = [
            MethodStyle(method, color=None, linestyle=None, label=method)
            for method in df.method.unique()
        ]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
    metrics = ["normalized_score", "rank"]
    for i, metric in enumerate(metrics):
        for j, method_style in enumerate(method_styles):
            xx = df.loc[df.method == method_style.name, metric].sort_values()
            if len(xx) > 0:
                axes[i].plot(
                    xx.values, np.arange(len(xx)) / len(xx),
                    label=method_style.name if method_style.label else None,
                    color=method_style.color,
                    linestyle=method_style.linestyle,
                    lw=1.5,
                )
                axes[i].set_title(metric.replace("_", "-"))
                axes[i].set_xlabel(metric.replace("_", "-"))
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
