from pathlib import Path

from tabrepo.paper.paper_runner_tabarena import PaperRunTabArena
from nips2025_utils.load_final_paper_results import load_paper_results
from autogluon.common.loaders import load_pd
import pandas as pd
from autogluon.common.utils.s3_utils import upload_s3_folder


def load_df_ensemble_weights(context_name: str) -> pd.DataFrame:
    df_ensemble_weights = load_pd.load(
        path=f"https://tabarena.s3.us-west-2.amazonaws.com/evaluation/{context_name}/data/df_portfolio_ensemble_weights.parquet"
    )

    framework_types = [
        "GBM",
        "XGB",
        "CAT",
        "NN_TORCH",
        "FASTAI",
        "KNN",
        "RF",
        "XT",
        "LR",
        "TABPFNV2",
        "TABICL",
        "TABDPT",
        "REALMLP",
        "EBM",
        "FT_TRANSFORMER",
        "TABM",
        "MNCA",
    ]

    from tabrepo.paper.paper_utils import get_framework_type_method_names
    f_map, f_map_type, f_map_inverse, f_map_type_name = get_framework_type_method_names(
        framework_types=framework_types,
        max_runtimes=[
            (3600 * 4, "_4h"),
            (None, None),
        ]
    )
    df_ensemble_weights = df_ensemble_weights.rename(columns=f_map_type_name)
    return df_ensemble_weights



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
    eval_save_path = Path(context_name) / "output"
    load_from_s3 = False  # Do this for first run, then make false for speed
    generate_from_repo = False
    with_holdout = True
    elo_bootstrap_rounds = 10

    print(f"Loading results... context_name={context_name}, load_from_s3={load_from_s3}")
    df_results, df_results_holdout, datasets_tabpfn, datasets_tabicl = load_paper_results(
        context_name=context_name,
        generate_from_repo=generate_from_repo,
        load_from_s3=load_from_s3,
        # save_local_to_s3=True,
    )

    eval_save_path_w_holdout = eval_save_path / f"with_holdout"
    eval_save_path_w_portfolio = eval_save_path / f"with_portfolio"

    df_ensemble_weights = load_df_ensemble_weights(context_name=context_name)
    # paper_full.plot_ensemble_weights_heatmap(
    #     df_ensemble_weights=df_ensemble_weights,
    #     figsize=(24, 20),
    # )

    load_holdout = False

    df_results = df_results[~df_results["framework"].isin(banned_methods)]

    print(f"Starting evaluations...")

    # plots for sub-benchmarks, with and without imputation
    for use_tabpfn in [False, True]:
        for use_tabicl in [False, True]:
            for use_imputation in [False, True]:
                folder_name = ("tabpfn-tabicl" if use_tabpfn else "tabicl") \
                    if use_tabicl else ("tabpfn" if use_tabpfn else "full")
                if use_imputation:
                    folder_name = folder_name + "-imputed"

                banned_model_types = []
                imputed_models = []
                if not use_tabicl:
                    banned_model_types.append("TABICL")
                    imputed_models.append("TabICL")
                if not use_tabpfn:
                    banned_model_types.append("TABPFNV2")
                    imputed_models.append("TabPFNv2")

                datasets = (list(set(datasets_tabpfn).intersection(datasets_tabicl)) if use_tabpfn else datasets_tabicl) \
                    if use_tabicl else (datasets_tabpfn if use_tabpfn else None)

                plotter = PaperRunTabArena(
                    repo=None,
                    output_dir=eval_save_path / folder_name,
                    datasets=datasets,
                    banned_model_types=None if use_imputation else banned_model_types,
                    elo_bootstrap_rounds=elo_bootstrap_rounds,
                )

                if not use_tabpfn and not use_tabicl:
                    plotter.plot_portfolio_ensemble_weights_barplot(
                        df_ensemble_weights=df_ensemble_weights,
                    )

                plotter.eval(df_results=df_results, imputed_names=imputed_models,
                    only_datasets_for_method={'TabPFNv2': datasets_tabpfn, 'TabICL': datasets_tabicl})


    # plots for binary, regression, multiclass
    problem_types = ["binary", "regression", "multiclass"]
    for problem_type in problem_types:
        eval_save_path_problem_type = eval_save_path / problem_type
        paper_run_problem_type = PaperRunTabArena(
            repo=None,
            output_dir=eval_save_path_problem_type,
            problem_types=[problem_type],
            elo_bootstrap_rounds=elo_bootstrap_rounds,
            # banned_model_types=[
            #     "TABPFNV2",
            #     "TABICL",
            # ],
        )
        paper_run_problem_type.eval(
            df_results=df_results,
            imputed_names=['TabPFNv2', 'TabICL'],
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
            "tab:purple",
        ]
    )

    # upload_results(folder_to_upload=eval_save_path, s3_prefix=eval_save_path)
