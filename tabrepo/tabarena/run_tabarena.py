from __future__ import annotations

from autogluon.common.loaders import load_pd
from autogluon.common.savers import save_pd
from tabrepo.tabarena.tabarena import TabArena


if __name__ == '__main__':
    context_name = "D244_F3_C1530_10"
    save_path = f"../paper/tmp/{context_name}/df_results.parquet"
    df_results = load_pd.load(path=save_path)
    df_results = df_results.reset_index(drop=False)
    df_results = df_results.drop(columns=["problem_type"])
    df_results = df_results.drop(columns=["seed"])
    df_results = df_results.drop(columns=["metric"])
    # df_results = df_results.drop(columns=["fold"])

    df_results = df_results.rename(columns={"framework": "method"})
    df_results = df_results.rename(columns={"dataset": "task"})
    df_results = df_results.rename(columns={"metric_error": "err"})
    arena = TabArena(
        method_col="method",
        task_col="task",
        error_col="err",
        groupby_columns=[
            # "problem_type"
        ],
        seed_column="fold",
    )
    results = arena.leaderboard(
        data=df_results,
        include_error=True,
        include_elo=True,
        include_failure_counts=True,
        include_mrr=True,
        include_rank_counts=True,
        include_winrate=True,
    )
    print(results)

    out_path = "demo_out.parquet"
    # save_pd.save(path=out_path, df=results)
    results_load = load_pd.load(path=out_path)

    out = results.equals(results_load)
    print(out)
