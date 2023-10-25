import copy

import pandas as pd

from tabrepo.repository.evaluation_repository import EvaluationRepository

from scripts.baseline_comparison.plot_utils import save_latex_table


def winrate_comparison(df: pd.DataFrame, repo: EvaluationRepository):
    # git clone https://github.com/Innixma/autogluon-benchmark.git
    # cd autogluon-benchmark
    # pip install -e .
    from autogluon_benchmark.evaluation.evaluate_results import evaluate

    df = df.rename({
        "time fit (s)": "time_train_s",
        "time infer (s)": "time_infer_s",
    }, axis=1)

    df_for_eval = copy.deepcopy(df)
    df_for_eval = df_for_eval.rename(columns={
        'method': 'framework',
        'test_error': 'metric_error',
    })

    tid_to_dataset = repo._tid_to_dataset_dict
    df_for_eval['dataset'] = df['tid'].map(tid_to_dataset)
    df_for_eval['problem_type'] = df_for_eval['dataset'].map(repo._zeroshot_context.dataset_to_problem_type_dict)

    df_for_eval = df_for_eval.drop(columns=['rank', 'normalized-error'])

    # from autogluon.common.savers import save_pd
    # save_path = f's3://autogluon-zeroshot/config_results/zs_Bag244_full.csv'
    # save_pd.save(path=save_path, df=df_for_eval)

    frameworks_to_eval = [
        "Portfolio-N200 (ensemble)",
        "AutoGluon best",
        "Lightautoml",
        "Autosklearn2",
        "CatBoost (tuned + ensemble)",
        "Flaml",
        "H2oautoml",
        "Autosklearn",
    ]

    for time_limit in ["1h", "4h"]:
        frameworks_to_eval_time_limit = [f + f' ({time_limit})' for f in frameworks_to_eval]
        results_ranked_valid, results_ranked_by_dataset_valid, results_ranked_all, results_ranked_by_dataset_all, results_pairs_merged_dict = evaluate(
            results_raw=df_for_eval, frameworks=frameworks_to_eval_time_limit, frameworks_compare_vs_all=[frameworks_to_eval_time_limit[0]],
            columns_to_agg_extra=['time_infer_s']
        )
        results_pairs_portfolio = results_pairs_merged_dict[frameworks_to_eval_time_limit[0]]
        results_pairs_portfolio_min = results_pairs_portfolio[[
            "framework", "winrate", ">", "<", "=", "time_train_s", "time_infer_s", "loss_rescaled", "rank",
            # "rank=1_count", "rank=2_count", "rank=3_count", "rank>3_count",
        ]]

        results_pairs_portfolio_min = results_pairs_portfolio_min.rename({
            "time_train_s": "time fit (s)",
            "time_infer_s": "time infer (s)",
            "loss_rescaled": "loss (rescaled)",
            "framework": "method",
            # ">": "$>$",
            # "<": "$<$",
        },
            axis=1,
        )

        latex_kwargs = dict(index=False)
        n_digits = {
            "winrate": 3,
            "time fit (s)": 1,
            "time infer (s)": 3,
            "loss (rescaled)": 3,
            "rank": 3,
        }
        save_latex_table(df=results_pairs_portfolio_min, title=f"portfolio_winrate_compare_{time_limit}", show_table=True,
                         latex_kwargs=latex_kwargs, n_digits=n_digits)
