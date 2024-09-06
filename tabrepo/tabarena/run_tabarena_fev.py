from __future__ import annotations

import pandas as pd

from autogluon.tabular import TabularDataset
from tabrepo.tabarena.tabarena import TabArena


if __name__ == '__main__':
    fev_prefix = "../../../fev/benchmarks/chronos_zeroshot/results/"

    results_files = [
        "auto_arima.csv",
        "auto_ets.csv",
        "auto_theta.csv",
        "chronos_base.csv",
        "chronos_bolt_base.csv",
        "chronos_bolt_mini.csv",
        "chronos_bolt_small.csv",
        "chronos_bolt_tiny.csv",
        "chronos_large.csv",
        "chronos_mini.csv",
        "chronos_small.csv",
        "chronos_tiny.csv",
        "moirai_base.csv",
        "moirai_large.csv",
        "moirai_small.csv",
        "seasonal_naive.csv",
        "timesfm-2.0.csv",
        "timesfm.csv",
    ]

    results_files = [f"{fev_prefix}{result_file}" for result_file in results_files]

    results_dfs = [
        TabularDataset(result_file) for result_file in results_files
    ]

    df_results = pd.concat(results_dfs, ignore_index=True)

    arena = TabArena(
        method_col="model_name",
        task_col="dataset_name",
        error_col="WQL",
        columns_to_agg_extra=[
            # "training_time",
            "inference_time_s",
        ],
    )

    results_agg = arena.leaderboard(
        data=df_results,
        # include_error=True,
        include_elo=True,
        include_failure_counts=True,
        include_mrr=True,
        include_rank_counts=True,
        include_winrate=True,
        elo_kwargs={
            # "calibration_framework": "Median",
            "calibration_elo": 1000,
            "BOOTSTRAP_ROUNDS": 1000,
        },
        baseline_relative_error="seasonal_naive",
        relative_error_kwargs={"agg": "gmean"},
    )
    print(results_agg)

    results_per_task = arena.compute_results_per_task(data=df_results)
    arena.plot_critical_diagrams(results_per_task=results_per_task, save_path=f"./figures/fev/critical-diagram.png")
    arena.plot_critical_diagrams(results_per_task=results_per_task, save_path=f"./figures/fev/critical-diagram.pdf")

    results_per_task_rename = results_per_task.rename(columns={
        arena.method_col: "framework",
        arena.task_col: "dataset",
        # "training_time": "time_train_s",
        "inference_time_s": "time_infer_s",
        arena.error_col: "metric_error",
    })
    # FIXME: Don't require time_train_s, time_infer_s
    results_per_task_rename["time_train_s"] = results_per_task_rename["time_infer_s"]

    from autogluon_benchmark.plotting.plotter import Plotter
    plotter = Plotter(
        results_ranked_df=results_per_task_rename,
        results_ranked_fillna_df=results_per_task_rename,
        save_dir=f"./figures/fev"
    )

    plotter.plot_all(
        calibration_framework="seasonal_naive",
        calibration_elo=1000,
    )
