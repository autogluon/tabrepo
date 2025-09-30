from __future__ import annotations

from dataclasses import dataclass
from scripts import output_path

import matplotlib.pyplot as plt
from typing import List
import numpy as np
import pandas as pd


@dataclass
class MethodStyle:
    name: str
    color: str
    linestyle: str = None  # linestyle of the method, default to plain
    linewidth: float = None
    label: bool = True  # whether to show the method name as label
    label_str: str = None


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
                df[col] = df[col].astype("object")
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
