from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
from tabrepo.utils.normalized_scorer import NormalizedScorer
from tabrepo.utils.rank_utils import RankScorer
from tabrepo import EvaluationRepository
from scripts.baseline_comparison.plot_utils import (
    figure_path,
)
from scripts.baseline_comparison.baselines import zeroshot_name

default_ensemble_size = 40


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
    tasks = repo.tasks()
    rank_scorer = RankScorer(df_results_baselines, tasks=unique_dataset_folds, pct=False)
    normalized_scorer = NormalizedScorer(df_results_baselines, tasks=tasks, baseline=None)
    return rank_scorer, normalized_scorer


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


def framework_name(framework_type, max_runtime=None, ensemble_size=default_ensemble_size, tuned: bool=True, all: bool = False, prefix: str = None) -> str:
    method = framework_type if framework_type else "All"
    if prefix is None:
        prefix = ""
    if all:
        method = "All"
    if not tuned:
        suffix = " (default)"
    else:
        suffix = " (tuned + ensemble)" if ensemble_size > 1 else " (tuned)"
        suffix += time_suffix(max_runtime=max_runtime)
    method = f"{method}{prefix}{suffix}"
    return method


def time_suffix(max_runtime: float) -> str:
    if max_runtime:
        if max_runtime >= 3600:
            str_num_hours = f"{int(max_runtime / 3600)}" if max_runtime % 3600 == 0 else f"{max_runtime / 3600:0.2f}"
            return f" ({str_num_hours}h)"
        else:
            str_num_mins = f"{int(max_runtime / 60)}" if max_runtime % 60 == 0 else f"{max_runtime / 60:0.2f}"
            return f" ({str_num_mins}m)"
    else:
        return ""


def generate_sensitivity_plots(df, n_portfolios: list = None, n_ensemble_iterations: list = None, show: bool = False, save_prefix: str = None):
    plt.rcParams.update({'font.size': 11})
    fig, axes = plt.subplots(1, 4, sharey=True, sharex=True, figsize=(11, 2.2))

    dimensions = [
        ("M", "#configurations per family"),
        ("D", "#training datasets"),
    ]

    metric = "normalized-error"

    baselines = ["Autosklearn2 (4h)", "AutoGluon 0.8 (4h)", "AutoGluon 1.1 (4h)", ]
    baseline_colors = ["darkgray", "black", "blue"]
    baseline_errors = [df.loc[df.method.str.contains(baseline, regex=False), metric].mean() for baseline in baselines]

    def decorate(ax, add_ylabel):
        ax.set_xlim([0, 200])
        ax.set_xlabel(legend)
        if add_ylabel:
            ax.set_ylabel(f"{metric}")
        ax.grid()
        for baseline, color, mean_err in zip(baselines, baseline_colors, baseline_errors):
            ax.hlines(mean_err, xmin=0, xmax=200, color=color, label=baseline, ls="--")

    for i, (dimension, legend) in enumerate(dimensions):
        regex = f"Portfolio-N.*-{dimension}.*4h"
        df_portfolio = df.loc[df.method.str.contains(regex), :].copy()
        is_ensemble = df_portfolio.method.str.contains("(ensemble)", regex=False)
        df_portfolio.loc[is_ensemble, dimension] = df_portfolio.loc[is_ensemble, "method"].apply(
            lambda s: int(s.replace(" (ensemble) (4h)", "").split("-")[-1][1:]))
        df_portfolio.loc[~is_ensemble, dimension] = df_portfolio.loc[~is_ensemble, "method"].apply(
            lambda s: int(s.replace(" (4h)", "").split("-")[-1][1:]))

        # Drop instances where dimension=1
        df_portfolio = df_portfolio.loc[df_portfolio[dimension] != 1, :]

        for is_ens in [False, True]:
            df_portfolio_agg = df_portfolio.loc[is_ensemble] if is_ens else df_portfolio.loc[~is_ensemble]
            df_portfolio_agg = df_portfolio_agg[[dimension, metric, "seed"]].groupby([dimension, "seed"]).mean()[
                metric]
            dim, mean, sem = df_portfolio_agg.groupby(dimension).agg(["mean", "sem"]).reset_index().values.T

            ax = axes[i]
            label = "Portfolio"
            if is_ens:
                label += " (ens.)"

            ax.plot(
                dim, mean,
                label=label,
                linestyle="-",
                marker="o",
                linewidth=0.6,
                markersize=4,
            )

            # ax.fill_between(
            #     dim,
            #     [m - s for m, s in zip(mean, sem)],
            #     [m + s for m, s in zip(mean, sem)],
            #     alpha=0.2,
            # )
        decorate(ax, add_ylabel=i==0)

    # dictionary to extract the number of portfolio member from names
    extract_number_portfolio = {
        zeroshot_name(n_portfolio=size): size
        for i, size in enumerate(n_portfolios)
    }

    extract_number_portfolio.update(
        {
            zeroshot_name(n_portfolio=size, n_ensemble=1): size
            for i, size in enumerate(n_portfolios)
        }
    )

    df["N"] = df["method"].map(extract_number_portfolio)

    # dictionary to extract the number of ensemble members from names
    extract_number_ensemble = {
        zeroshot_name(n_ensemble=size, n_ensemble_in_name=True): size
        for i, size in enumerate(n_ensemble_iterations)
    }
    df["C"] = df["method"].map(extract_number_ensemble)

    dimensions = [
        ("N", "#portfolio configurations"),
        ("C", "#ensemble members"),
    ]

    for i, (dimension, legend) in enumerate(dimensions):
        ax = axes[i + 2]

        df_portfolio = df.loc[~df[dimension].isna()]

        # Drop instances where dimension=1
        df_portfolio = df_portfolio.loc[df_portfolio[dimension] != 1, :]

        is_ensemble = df_portfolio.method.str.contains("(ensemble)", regex=False)

        for is_ens in [False, True]:
            df_portfolio_agg = df_portfolio.loc[is_ensemble] if is_ens else df_portfolio.loc[~is_ensemble]
            df_portfolio_agg = df_portfolio_agg[[dimension, metric]].groupby(dimension).mean()[metric]
            dim, mean = df_portfolio_agg.reset_index().values.T

            label = "Portfolio"
            if is_ens:
                label += " (ens.)"

            ax.plot(
                dim, mean,
                label=label,
                linestyle="-",
                marker="o",
                linewidth=0.6,
                markersize=4,
            )
        decorate(ax, add_ylabel=False)
    axes[-1].legend()

    fig_path = figure_path(prefix=save_prefix)
    fig_save_path = fig_path / f"sensitivity.png"
    plt.tight_layout()
    plt.savefig(fig_save_path)
    if show:
        plt.show()
