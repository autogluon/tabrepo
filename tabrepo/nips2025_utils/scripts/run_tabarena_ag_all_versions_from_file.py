from __future__ import annotations

from autogluon.common.loaders import load_pd
from tabrepo.tabarena.tabarena import TabArena
from autogluon_benchmark.plotting.plotter import Plotter


if __name__ == '__main__':
    df_results = load_pd.load(path="s3://neerick-autogluon/tabarena/benchmarks/amlb2025_all/results.csv")
    baseline_method = "RandomForest_60min"

    arena = TabArena(
        method_col="framework",
        task_col="task",
        error_col="metric_error",
        columns_to_agg_extra=[
            "training_duration",
            "predict_duration",
        ],
        seed_column="fold",
    )

    df_results_fillna = arena.fillna_data(data=df_results, fillna_method="worst")

    results_agg = arena.leaderboard(
        data=df_results_fillna,
        # include_error=True,
        include_elo=True,
        include_failure_counts=True,
        include_mrr=True,
        include_rank_counts=True,
        include_winrate=True,
        elo_kwargs={
            "calibration_framework": baseline_method,
            "calibration_elo": 1000,
            "BOOTSTRAP_ROUNDS": 100,
        },
        baseline_relative_error=baseline_method,
        relative_error_kwargs={"agg": "gmean"},
    )
    print(results_agg)

    results_per_task = arena.compute_results_per_task(data=df_results_fillna)
    arena.plot_critical_diagrams(results_per_task=results_per_task, save_path=f"./figures/autogluon2025/critical-diagram.png")
    arena.plot_critical_diagrams(results_per_task=results_per_task, save_path=f"./figures/autogluon2025/critical-diagram.pdf")

    results_per_task_rename = results_per_task.rename(columns={
        arena.method_col: "framework",
        arena.task_col: "dataset",
        "training_duration": "time_train_s",
        "predict_duration": "time_infer_s",
        arena.error_col: "metric_error",
    })

    plotter = Plotter(
        results_ranked_df=results_per_task_rename,
        results_ranked_fillna_df=results_per_task_rename,
        save_dir=f"./figures/autogluon2025"
    )

    plotter.plot_all(
        calibration_framework=baseline_method,
        calibration_elo=1000,
    )
