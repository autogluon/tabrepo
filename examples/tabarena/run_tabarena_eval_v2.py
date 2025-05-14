from tabrepo.paper.paper_runner_tabarena import PaperRunTabArena
from nips2025_utils.load_final_paper_results import load_paper_results
from autogluon.common.loaders import load_pd
import pandas as pd


"""
Ensure your IDE/env recognizes the `tabrepo/scripts` folder as `from scripts import ...`,
otherwise the below code will fail.

This script loads the complete results data and runs evaluation on it, creating plots and tables.
"""


if __name__ == '__main__':
    context_name = "tabarena_paper_full_gpu"
    eval_save_path = f"{context_name}/output"
    load_from_s3 = False  # Do this for first run, then make false for speed
    generate_from_repo = False
    with_holdout = True
    ban_datasets = True

    print(f"Loading results... context_name={context_name}, load_from_s3={load_from_s3}")
    df_results, datasets_tabpfn, datasets_tabicl = load_paper_results(
        context_name=context_name,
        generate_from_repo=generate_from_repo,
        load_from_s3=load_from_s3,
    )

    if with_holdout:
        eval_save_path = f"{eval_save_path}_w_holdout"
    if ban_datasets:
        eval_save_path = f"{eval_save_path}_51"

    eval_save_path_tabpfn_datasets = f"{eval_save_path}/tabpfn_datasets"
    eval_save_path_tabicl_datasets = f"{eval_save_path}/tabicl_datasets"
    eval_save_path_full = f"{eval_save_path}/full"
    paper_full = PaperRunTabArena(repo=None, output_dir=eval_save_path_full)

    if with_holdout:
        from tabrepo import EvaluationRepositoryCollection, Evaluator
        repo_holdout = EvaluationRepositoryCollection.load(path=f"./{context_name}/repo_cache/tabarena_holdout.pkl")
        evaluator_holdout = Evaluator(repo_holdout)

        repo_holdout.set_config_fallback("RandomForest_r1_BAG_L1_HOLDOUT")

        # FIXME: Fillna
        # df_results_holdout = evaluator_holdout.compare_metrics(keep_extra_columns=True, include_metric_error_val=True, fillna=True).reset_index()
        df_results_holdout = load_pd.load(path=f"{context_name}/data/df_results_holdout.parquet")

        # FIXME
        df_results = pd.concat([df_results, df_results_holdout], ignore_index=True)

        # FIXME
        df_results = df_results.drop_duplicates(subset=[
            "dataset",
            "fold",
            "framework",
            "seed",
        ], ignore_index=True)

        df_results = paper_full.compute_normalized_error(df_results=df_results)

        # FIXME
        raise AssertionError

    banned_methods = [
        "Portfolio-N20 (ensemble) (4h)",
        "Portfolio-N10 (ensemble) (4h)",
        "Portfolio-N5 (ensemble) (4h)",
        "Portfolio-N50 (4h)",
        "Portfolio-N20 (4h)",
        "Portfolio-N10 (4h)",
        "Portfolio-N5 (4h)",
        "AutoGluon 1.3 (1h)",
    ]
    df_results = df_results[~df_results["framework"].isin(banned_methods)]

    banned_datasets = [
        "ASP-POTASSCO",
        "Mobile_Price",
        "Pumpkin_Seeds",
        "abalone",
        "fifa",
        "internet_firewall",
        "cars",
        "steel-plates-fault",
        "solar_flare",
        "PhishingWebsites",
    ]
    df_results = df_results[~df_results["dataset"].isin(banned_datasets)]

    df_results = paper_full.compute_normalized_error(df_results=df_results)

    paper_tabpfn_datasets = PaperRunTabArena(repo=None, output_dir=eval_save_path_tabpfn_datasets, datasets=datasets_tabpfn)
    paper_tabicl_datasets = PaperRunTabArena(repo=None, output_dir=eval_save_path_tabicl_datasets, datasets=datasets_tabicl)

    print(f"Starting evaluations...")
    # Full run
    paper_full.eval(df_results=df_results)

    problem_types = ["binary", "regression", "multiclass"]
    for problem_type in problem_types:
        eval_save_path_problem_type = f"{eval_save_path}/{problem_type}"
        paper_run_problem_type = PaperRunTabArena(repo=None, output_dir=eval_save_path_problem_type, problem_types=[problem_type])
        paper_run_problem_type.eval(df_results=df_results)

    # Only TabPFN datasets
    paper_tabpfn_datasets.eval(df_results=df_results)

    # Only TabICL datasets
    paper_tabicl_datasets.eval(df_results=df_results)
