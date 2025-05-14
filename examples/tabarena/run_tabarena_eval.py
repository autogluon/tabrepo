from tabrepo.paper.paper_runner_tabarena import PaperRunTabArena
from nips2025_utils.load_final_paper_results import load_paper_results


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

    print(f"Loading results... context_name={context_name}, load_from_s3={load_from_s3}")
    df_results, datasets_tabpfn, datasets_tabicl = load_paper_results(
        context_name=context_name,
        generate_from_repo=generate_from_repo,
        load_from_s3=load_from_s3,
    )

    eval_save_path_tabpfn_datasets = f"{eval_save_path}/tabpfn_datasets"
    eval_save_path_tabicl_datasets = f"{eval_save_path}/tabicl_datasets"

    paper_full = PaperRunTabArena(repo=None, output_dir=eval_save_path)
    paper_tabpfn_datasets = PaperRunTabArena(repo=None, output_dir=eval_save_path_tabpfn_datasets, datasets=datasets_tabpfn)
    paper_tabicl_datasets = PaperRunTabArena(repo=None, output_dir=eval_save_path_tabicl_datasets, datasets=datasets_tabicl)

    print(f"Starting evaluations...")
    # Full run
    paper_full.eval(df_results=df_results, only_norm_scores=False, imputed_names=['TabPFNv2', 'TabICL'])

    problem_types = ["binary", "regression", "multiclass"]
    for problem_type in problem_types:
        eval_save_path_problem_type = f"{eval_save_path}/{problem_type}"
        paper_run_problem_type = PaperRunTabArena(repo=None, output_dir=eval_save_path_problem_type, problem_types=[problem_type])
        paper_run_problem_type.eval(df_results=df_results, imputed_names=['TabPFNv2', 'TabICL'])

    # Only TabPFN datasets
    paper_tabpfn_datasets.eval(df_results=df_results, imputed_names=['TabICL'])

    # Only TabICL datasets
    paper_tabicl_datasets.eval(df_results=df_results, imputed_names=['TabPFNv2'])
