from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from tabrepo.nips2025_utils.tabarena_context import TabArenaContext
from tabrepo.tabarena.tabarena import TabArena
from autogluon.common.loaders import load_pd

from tabrepo.paper.paper_utils import get_method_rename_map
from tabrepo.nips2025_utils.artifacts._tabarena_method_metadata import tabarena_method_metadata_2025_06_12_collection_main
from tabrepo.plot.plot_pareto_frontier import plot_optimal_arrow


def plot_hpo(
    df: pd.DataFrame,
    xlabel: str,
    ylabel: str,
    save_path: str,
    higher_is_better: bool = True,
    method_col: str = "name",
    xlog: bool = True,
    color_by_rank: bool = True,
):
    # Build a 60-color palette from tab20 / tab20b / tab20c
    colors60 = (
        list(sns.color_palette("tab20", 20)) +
        list(sns.color_palette("tab20b", 20)) +
        list(sns.color_palette("tab20c", 20))
    )

    method_names = list(df[method_col].unique())

    # Determine peak per method (max if higher_is_better else min)
    if higher_is_better:
        peak_per_method = {
            m: df.loc[df[method_col] == m, ylabel].max()
            for m in method_names
        }
    else:
        peak_per_method = {
            m: df.loc[df[method_col] == m, ylabel].min()
            for m in method_names
        }

    # Sort by peak and create a stable color map (alphabetical) so colors don't change with sort order
    sorted_methods = sorted(method_names, key=lambda m: peak_per_method[m], reverse=higher_is_better)
    if color_by_rank:
        base_methods_for_colors = sorted_methods
    else:
        base_methods_for_colors = sorted(method_names)
    color_map = {m: colors60[i % len(colors60)] for i, m in enumerate(base_methods_for_colors)}

    fig, ax = plt.subplots(figsize=(8, 6))
    if xlog:
        ax.set_xscale("log")

    plot_optimal_arrow(ax=ax, max_X=False, max_Y=higher_is_better, size=0.6)

    handles = []
    labels = []
    for method_name in sorted_methods:
        df_method = df[df[method_col] == method_name]
        if df_method.empty:
            continue
        scores = df_method[ylabel].to_numpy()
        times = df_method[xlabel].to_numpy()
        h, = ax.plot(
            times,
            scores,
            ".-",
            label=method_name,
            color=color_map[method_name],
            markersize=16,
            alpha=0.8,
        )
        handles.append(h)
        labels.append(method_name)

    # Flip legend order only if higher_is_better is False
    if higher_is_better:
        ax.legend(handles, labels)
    else:
        ax.legend(handles[::-1], labels[::-1])

    ax.grid(True)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(save_path)


if __name__ == '__main__':
    include_portfolio = False
    include_hpo_seeds = False
    average_seeds = True

    method_rename_map = get_method_rename_map()
    method_rename_map["REALMLP"] = "RealMLP"
    framework_types = list(method_rename_map.keys())

    # results_file = "hpo_new_lb.parquet"
    # s3_path = "s3://tabarena/tmp/camera_ready/hpo_new_lb.parquet"

    results_file = "hpo_camera_ready_lb.parquet"
    s3_path = "s3://tabarena/tmp/camera_ready/hpo_camera_ready_lb.parquet"
    from autogluon.common.savers import save_pd

    try:
        results_hpo = load_pd.load(path=results_file)
    except:
        print(f"Downloading from s3: {s3_path}")
        # download from s3 if missing
        results_hpo = load_pd.load(path=s3_path)
        save_pd.save(path=results_file, df=results_hpo)
        results_hpo = load_pd.load(path=results_file)
    # save_pd.save(path=s3_path, df=results_hpo)

    use_old_lb = True
    if use_old_lb:
        methods_lst = [m for m in tabarena_method_metadata_2025_06_12_collection_main.method_metadata_lst if m.method_type != "portfolio"]
        tabarena_context = TabArenaContext(methods=methods_lst)
    else:
        tabarena_context = TabArenaContext()

    calibration_framework = "RF (default)"
    elo_bootstrap_rounds = 1

    result_baselines = tabarena_context.load_results_paper()
    task_metadata = tabarena_context.task_metadata

    results_hpo_mean = results_hpo.copy().groupby(["method", "dataset", "fold", "problem_type", "metric", "config_type"]).mean(
        numeric_only=True
    ).drop(columns=["seed"]).reset_index()

    results_lst = [
        results_hpo_mean,
    ]

    if include_hpo_seeds:
        results_hpo_seeds = results_hpo.copy()
        results_hpo_seeds["method"] = results_hpo_seeds["method"] + "-" + results_hpo_seeds["seed"].astype(str)
        results_hpo_seeds["config_type"] = results_hpo_seeds["config_type"] + "-" + results_hpo_seeds["seed"].astype(str)
        results_hpo_seeds = results_hpo_seeds.groupby(["method", "dataset", "fold", "problem_type", "metric", "config_type"]).mean(
            numeric_only=True).reset_index()
        results_lst.append(results_hpo_seeds)

    if include_portfolio:
        results_portfolio = load_pd.load(path="rebuttal_portfolio_n_configs.parquet")
        results_portfolio["config_type"] = results_portfolio["method"]
        results_portfolio["method"] = results_portfolio["method"] + "-" + results_portfolio["n_portfolio"].astype(str)
        results_lst.append(results_portfolio)

    results_hpo = pd.concat(results_lst, ignore_index=True)
    combined_data = pd.concat([result_baselines, results_hpo], ignore_index=True)

    # ----- add times per 1K samples -----
    dataset_to_n_samples_train = tabarena_context.task_metadata.set_index("name")["n_samples_train_per_fold"].to_dict()
    dataset_to_n_samples_test = tabarena_context.task_metadata.set_index("name")["n_samples_test_per_fold"].to_dict()

    combined_data['time_train_s_per_1K'] = combined_data['time_train_s'] * 1000 / combined_data["dataset"].map(
        dataset_to_n_samples_train)
    combined_data['time_infer_s_per_1K'] = combined_data['time_infer_s'] * 1000 / combined_data["dataset"].map(
        dataset_to_n_samples_test)

    tabarena_init_kwargs = dict(
        task_col="dataset",
        columns_to_agg_extra=[
            "time_train_s",
            "time_infer_s",
            "time_train_s_per_1K",
            "time_infer_s_per_1K",
        ],
        groupby_columns=["problem_type", "metric"],
        seed_column="fold",
    )

    arena = TabArena(
        **tabarena_init_kwargs,
        error_col="metric_error",
    )

    arena_val = TabArena(
        **tabarena_init_kwargs,
        error_col="metric_error_val",
    )

    combined_data = combined_data[~combined_data["metric_error_val"].isna()]

    combined_data = arena.fillna_data(
        data=combined_data,
        fillna_method=calibration_framework,
    )

    results_per_task = arena.compute_results_per_task(data=combined_data)

    leaderboard = arena.leaderboard(
        data=combined_data,
        include_elo=True,
        elo_kwargs=dict(
            calibration_framework=calibration_framework,
            calibration_elo=1000,
            BOOTSTRAP_ROUNDS=elo_bootstrap_rounds,
        ),
        average_seeds=average_seeds,
        include_baseline_advantage=True,
    )

    leaderboard_val = arena_val.leaderboard(
        data=combined_data,
        include_elo=True,
        elo_kwargs=dict(
            calibration_framework=calibration_framework,
            calibration_elo=1000,
            BOOTSTRAP_ROUNDS=elo_bootstrap_rounds,
        ),
        average_seeds=average_seeds,
        include_baseline_advantage=True,
    )

    leaderboard["elo_val"] = leaderboard_val["elo"]
    leaderboard["improvability_val"] = leaderboard_val["improvability"]
    leaderboard["baseline_advantage_val"] = leaderboard_val["baseline_advantage"]

    leaderboard = leaderboard.reset_index(drop=False)

    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(leaderboard)

    methods_map = results_hpo[["method", "n_portfolio", "n_ensemble", "config_type"]].drop_duplicates(subset=["method"]).set_index("method")
    leaderboard = leaderboard[leaderboard["method"].isin(methods_map.index)]
    leaderboard["n_portfolio"] = leaderboard["method"].map(methods_map["n_portfolio"])
    leaderboard["config_type"] = leaderboard["method"].map(methods_map["config_type"])

    leaderboard["name"] = leaderboard["config_type"]

    leaderboard = leaderboard.sort_values(by=["config_type", "n_portfolio"])

    leaderboard["Elo"] = leaderboard["elo"]
    leaderboard["Elo (Val)"] = leaderboard["elo_val"]
    leaderboard["Improvability (%)"] = leaderboard["improvability"] * 100
    leaderboard["Improvability (%) (Val)"] = leaderboard["improvability_val"] * 100
    leaderboard["Improvability (%) (Test - Val)"] = (leaderboard["improvability"] - leaderboard["improvability_val"]) * 100

    leaderboard["Baseline Advantage (%) (Test)"] = leaderboard["baseline_advantage"] * 100
    leaderboard["Baseline Advantage (%) (Val)"] = leaderboard["baseline_advantage_val"] * 100
    leaderboard["Baseline Advantage (%) (Test - Val)"] = (leaderboard["baseline_advantage"] - leaderboard[
        "baseline_advantage_val"]) * 100

    leaderboard['Train time per 1K samples (s) (median)'] = leaderboard["median_time_train_s_per_1K"]
    leaderboard['Inference time per 1K samples (s) (median)'] = leaderboard["median_time_infer_s_per_1K"]

    leaderboard["name"] = leaderboard["name"].map(method_rename_map).fillna(leaderboard["name"])

    file_ext = ".pdf"

    ban_bad_methods = True
    if ban_bad_methods:
        bad_methods = ["KNN", "LR"]
        leaderboard = leaderboard[~leaderboard["config_type"].isin(bad_methods)]

    plot_hpo(
        df=leaderboard,
        xlabel="Train time per 1K samples (s) (median)",
        ylabel="Elo",
        save_path=f"pareto_n_configs_elo{file_ext}",
        higher_is_better=True,
    )
    plot_hpo(
        df=leaderboard,
        xlabel="Train time per 1K samples (s) (median)",
        ylabel="Improvability (%)",
        save_path=f"pareto_n_configs_imp{file_ext}",
        higher_is_better=False,
    )
    plot_hpo(
        df=leaderboard,
        xlabel="Inference time per 1K samples (s) (median)",
        ylabel="Elo",
        save_path=f"pareto_n_configs_elo_infer{file_ext}",
        higher_is_better=True,
    )
    plot_hpo(
        df=leaderboard,
        xlabel="Inference time per 1K samples (s) (median)",
        ylabel="Improvability (%)",
        save_path=f"pareto_n_configs_imp_infer{file_ext}",
        higher_is_better=False,
    )

    plot_hpo(
        df=leaderboard,
        xlabel="Train time per 1K samples (s) (median)",
        ylabel="Baseline Advantage (%) (Test - Val)",
        save_path=f"pareto_n_configs_adv{file_ext}",
        higher_is_better=False,
    )

    plot_hpo(
        df=leaderboard,
        xlabel="Inference time per 1K samples (s) (median)",
        ylabel="Baseline Advantage (%) (Test - Val)",
        save_path=f"pareto_n_configs_adv_infer{file_ext}",
        higher_is_better=False,
    )

    plot_hpo(
        df=leaderboard,
        xlabel="Baseline Advantage (%) (Val)",
        ylabel="Baseline Advantage (%) (Test)",
        save_path=f"pareto_n_configs_adv_vs{file_ext}",
        higher_is_better=True,
        xlog=False,
    )

    plot_hpo(
        df=leaderboard,
        xlabel="Improvability (%) (Val)",
        ylabel="Improvability (%)",
        save_path=f"pareto_n_configs_imp_vs{file_ext}",
        higher_is_better=False,
        xlog=False,
    )

    plot_hpo(
        df=leaderboard,
        xlabel="Elo (Val)",
        ylabel="Elo",
        save_path=f"pareto_n_configs_elo_vs{file_ext}",
        higher_is_better=True,
        xlog=False,
    )
