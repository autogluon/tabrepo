from __future__ import annotations

import pandas as pd

from autogluon.tabular import TabularDataset
from tabrepo.tabarena.tabarena import TabArena


def select_best_per_category(df, arena: TabArena):
    import pprint
    agg_df = arena.leaderboard(df)
    CATEGORIES = {
        "Median": ["SimpleAverage-median"],
        "Model selection": ["BestValidationModel"],
        "Performance-based average": ['PerformanceWeightedAverage-exp-lnorm', 'PerformanceWeightedAverage-inv-lnorm', 'PerformanceWeightedAverage-sqr-lnorm', 'PerformanceWeightedAverage-exp', 'PerformanceWeightedAverage-inv', 'PerformanceWeightedAverage-sqr', 'SimpleAverage-median', 'SimpleAverage-mean'],
        "Greedy ensemble selection": ["GreedyEnsemble-100"],
        "Linear model": [m for m in df.method.unique() if m.startswith("LinearEnsemble")],
        "Nonlinear model": [m for m in df.method.unique() if "GBM" in m or "REALMLP" in m],
        "Stacker model selection": ["Stacked-BestValidation"],
        "Multi-layer stacking": ["Stacked-GreedyEnsemble"],
    }

    best_method_per_category = {}
    for cat, methods in CATEGORIES.items():
        best_method_per_category[cat] = agg_df[agg_df.index.isin(methods)].sort_values("rank").index.values[0]
    pprint.pprint(best_method_per_category, sort_dicts=False)
    return best_method_per_category


def preprocess_ts_paper_results(df_results, arena: TabArena):
    best_method_per_category = select_best_per_category(df=df_results, arena=arena)
    df_results_filtered = df_results[df_results["method"].isin(best_method_per_category.values())].copy()
    best_method_per_category_inverse = {v: k for k, v in best_method_per_category.items()}
    df_results_filtered["method"] = df_results_filtered["method"].map(best_method_per_category_inverse)
    return df_results_filtered


if __name__ == '__main__':
    s3_bucket = "neerick-autogluon"
    s3_prefix = "2025_03_31_ts_stack_results/results/"
    s3_path_prefix = f"s3://{s3_bucket}/{s3_prefix}"

    arena = TabArena(
        method_col="method",
        task_col="dataset",
        error_col="test_loss",
        columns_to_agg_extra=[
            "training_time",
            "inference_time",
        ],
    )

    for metric in ["sql", "mase"]:
        results_path = f"{s3_path_prefix}full_scores_{metric}.csv"
        df_results: pd.DataFrame = TabularDataset(results_path)
        df_results = preprocess_ts_paper_results(df_results=df_results, arena=arena)

        results_agg = arena.leaderboard(
            data=df_results,
            # include_error=True,
            include_elo=True,
            # include_failure_counts=True,
            # include_mrr=True,
            include_rank_counts=True,
            # include_winrate=True,
            elo_kwargs={
                "calibration_framework": "Median",
                "calibration_elo": 1000,
                "BOOTSTRAP_ROUNDS": 1000,
            },
            baseline_relative_error="Median",
            relative_error_kwargs={"agg": "gmean"},
        )
        print(results_agg)

        results_per_task = arena.compute_results_per_task(data=df_results)
        arena.plot_critical_diagrams(results_per_task=results_per_task, save_path=f"./figures/critical-diagram-{metric}.png")
        arena.plot_critical_diagrams(results_per_task=results_per_task, save_path=f"./figures/critical-diagram-{metric}.pdf")

        results_per_task_rename = results_per_task.rename(columns={
            "method": "framework",
            "training_time": "time_train_s",
            "inference_time": "time_infer_s",
            "test_loss": "metric_error",
        })

        from autogluon_benchmark.plotting.plotter import Plotter
        plotter = Plotter(
            results_ranked_df=results_per_task_rename,
            results_ranked_fillna_df=results_per_task_rename,
            save_dir=f"./figures/{metric}"
        )

        plotter.plot_all()
