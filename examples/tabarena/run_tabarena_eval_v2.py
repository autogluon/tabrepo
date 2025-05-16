from tabrepo.paper.paper_runner_tabarena import PaperRunTabArena
from nips2025_utils.load_final_paper_results import load_paper_results
from autogluon.common.loaders import load_pd
import pandas as pd
from autogluon.common.utils.s3_utils import upload_s3_folder


def upload_results(folder_to_upload: str, s3_prefix: str):
    upload_s3_folder(
        bucket="tabarena",
        prefix=f"tmp_neurips2025/{s3_prefix}",
        folder_to_upload=folder_to_upload,
    )


"""
Ensure your IDE/env recognizes the `tabrepo/scripts` folder as `from scripts import ...`,
otherwise the below code will fail.

This script loads the complete results data and runs evaluation on it, creating plots and tables.
"""

banned_methods = [
    # "Portfolio-N200 (ensemble) (4h)",
    "Portfolio-N100 (ensemble) (4h)",
    "Portfolio-N50 (ensemble) (4h)",
    "Portfolio-N20 (ensemble) (4h)",
    "Portfolio-N10 (ensemble) (4h)",
    "Portfolio-N5 (ensemble) (4h)",
    "Portfolio-N200 (4h)",
    "Portfolio-N100 (4h)",
    "Portfolio-N50 (4h)",
    "Portfolio-N20 (4h)",
    "Portfolio-N10 (4h)",
    "Portfolio-N5 (4h)",

    "AutoGluon_bq_1h8c",
    "AutoGluon_bq_5m8c",

    # "Portfolio-N200 (ensemble, holdout) (4h)",
    "Portfolio-N100 (ensemble, holdout) (4h)",
    "Portfolio-N50 (ensemble, holdout) (4h)",
    "Portfolio-N20 (ensemble, holdout) (4h)",
    "Portfolio-N10 (ensemble, holdout) (4h)",
    "Portfolio-N5 (ensemble, holdout) (4h)",
    # "Portfolio-N200 (4h)",
    # "Portfolio-N100 (4h)",
    # "Portfolio-N50 (4h)",
    # "Portfolio-N20 (4h)",
    # "Portfolio-N10 (4h)",
    # "Portfolio-N5 (4h)",
]


def rename_func(name):
    if "(tuned)" in name:
        name = name.rsplit("(tuned)", 1)[0]
        name = f"{name}(tuned, holdout)"
    elif "(tuned + ensemble)" in name:
        name = name.rsplit("(tuned + ensemble)", 1)[0]
        name = f"{name}(tuned + ensemble, holdout)"
    elif "(ensemble) (4h)" in name:
        name = name.rsplit("(ensemble) (4h)", 1)[0]
        name = f"{name}(ensemble, holdout) (4h)"
    return name


if __name__ == '__main__':
    context_name = "tabarena_paper_full_51"
    eval_save_path = f"{context_name}/output"
    load_from_s3 = False  # Do this for first run, then make false for speed
    generate_from_repo = False
    with_holdout = True
    elo_bootstrap_rounds = 100

    print(f"Loading results... context_name={context_name}, load_from_s3={load_from_s3}")
    df_results, df_results_holdout, datasets_tabpfn, datasets_tabicl = load_paper_results(
        context_name=context_name,
        generate_from_repo=generate_from_repo,
        load_from_s3=load_from_s3,
        # save_local_to_s3=True,
    )

    eval_save_path_w_holdout = f"{eval_save_path}/with_holdout"
    eval_save_path_w_portfolio = f"{eval_save_path}/with_portfolio"

    eval_save_path_tabpfn_datasets = f"{eval_save_path}/tabpfn_datasets"
    eval_save_path_tabicl_datasets = f"{eval_save_path}/tabicl_datasets"
    eval_save_path_full = f"{eval_save_path}/full"
    paper_full = PaperRunTabArena(
        repo=None,
        output_dir=eval_save_path_full,
        banned_model_types=[
            "TABPFNV2",
            "TABICL",
        ],
        elo_bootstrap_rounds=elo_bootstrap_rounds,
    )

    # raise AssertionError

    load_holdout = False

    df_results = df_results[~df_results["framework"].isin(banned_methods)]

    paper_tabpfn_datasets = PaperRunTabArena(
        repo=None,
        output_dir=eval_save_path_tabpfn_datasets,
        datasets=datasets_tabpfn,
        banned_model_types=[
            "TABICL"
        ],
        elo_bootstrap_rounds=elo_bootstrap_rounds,
    )
    paper_tabicl_datasets = PaperRunTabArena(
        repo=None,
        output_dir=eval_save_path_tabicl_datasets,
        datasets=datasets_tabicl,
        banned_model_types=[
            "TABPFNV2"
        ],
        elo_bootstrap_rounds=elo_bootstrap_rounds,
    )

    df_ensemble_weights = load_pd.load(
        path=f"https://tabarena.s3.us-west-2.amazonaws.com/evaluation/{context_name}/data/df_portfolio_ensemble_weights.parquet"
    )
    paper_full.plot_ensemble_weights_heatmap(df_ensemble_weights=df_ensemble_weights)

    # plot_tabarena_times(df_results=df_results, output_dir=eval_save_path_full,
    #                     sub_benchmarks={'TabPFNv2': datasets_tabpfn, 'TabICL': datasets_tabicl})

    print(f"Starting evaluations...")
    # Full run
    paper_full.eval(df_results=df_results, imputed_names=['TabPFNv2', 'TabICL'],
                    only_datasets_for_method={'TabPFNv2': datasets_tabpfn, 'TabICL': datasets_tabicl})

    problem_types = ["binary", "regression", "multiclass"]
    for problem_type in problem_types:
        eval_save_path_problem_type = f"{eval_save_path}/{problem_type}"
        paper_run_problem_type = PaperRunTabArena(
            repo=None,
            output_dir=eval_save_path_problem_type,
            problem_types=[problem_type],
            elo_bootstrap_rounds=elo_bootstrap_rounds,
            banned_model_types=[
                "TABPFNV2",
                "TABICL",
            ],
        )
        paper_run_problem_type.eval(
            df_results=df_results,
            imputed_names=['TabPFNv2', 'TabICL'],
        )

    # Only TabPFN datasets
    paper_tabpfn_datasets.eval(
        df_results=df_results,
        imputed_names=['TabICL'],
    )

    # Only TabICL datasets
    paper_tabicl_datasets.eval(
        df_results=df_results,
        imputed_names=['TabPFNv2'],
    )

    if with_holdout:
        df_results_holdout["framework"] = [rename_func(f) for f in df_results_holdout["framework"]]

        df_results_combined_holdout = pd.concat([df_results, df_results_holdout], ignore_index=True)
        df_results_combined_holdout = df_results_combined_holdout.drop_duplicates(subset=[
            "dataset",
            "fold",
            "framework",
            "seed",
        ], ignore_index=True)
        df_results_combined_holdout = df_results_combined_holdout[~df_results_combined_holdout["framework"].isin(banned_methods)]

        # must recalculate normalized error after concat
        df_results_combined_holdout = PaperRunTabArena.compute_normalized_error_dynamic(df_results=df_results_combined_holdout)

        from autogluon.common.savers import save_pd
        from autogluon.common.loaders import load_pd
        save_pd.save(df=df_results_combined_holdout, path=f"{context_name}/data/df_results_combined_holdout.parquet")
        df_results_combined_holdout = load_pd.load(path=f"{context_name}/data/df_results_combined_holdout.parquet")

        paper_w_holdout = PaperRunTabArena(
            repo=None,
            output_dir=eval_save_path_w_holdout,
            elo_bootstrap_rounds=elo_bootstrap_rounds,
            banned_model_types=[
                "TABPFNV2",
                "TABICL",
                "TABDPT",
                "KNN",
            ],
        )

        # TODO: Add logic to specify baselines for holdout, generate separate plots for holdout comparisons
        paper_w_holdout.eval(
            df_results=df_results_combined_holdout,
            imputed_names=['TabPFNv2', 'TabICL'],
            plot_tune_types=["holdout_tuned_ensembled", "tuned_ensembled"],
        )

    paper_w_portfolio = PaperRunTabArena(
        repo=None,
        output_dir=eval_save_path_w_portfolio,
        elo_bootstrap_rounds=elo_bootstrap_rounds,
        banned_model_types=[
            "TABPFNV2",
            "TABICL",
        ],
    )

    paper_w_portfolio.eval(
        df_results=df_results,
        baselines=[
            "AutoGluon 1.3 (4h)",
            "Portfolio-N200 (ensemble) (4h)",
        ],
        baseline_colors=[
            "black",
            "orange",
        ]
    )

    # upload_results(folder_to_upload=eval_save_path, s3_prefix=eval_save_path)
