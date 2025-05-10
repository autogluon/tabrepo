from autogluon.common.loaders import load_pd
from tabrepo.paper.paper_runner_tabarena import PaperRunTabArena


"""
Ensure your IDE/env recognizes the `tabrepo/scripts` folder as `from scripts import ...`,
otherwise the below code will fail.

This script loads the complete results data and runs evaluation on it, creating plots and tables.
"""

if __name__ == '__main__':
    context_name = "tabarena_paper"
    # df_result_save_path_w_norm_err = f"./{context_name}/data/df_results_w_norm_err.parquet"  # load from local
    # df_result_save_path_w_norm_err = f"s3://tabarena/evaluation/{context_name}/data/df_results_w_norm_err.parquet"  # load from s3 (private)
    df_result_save_path_w_norm_err = f"https://tabarena.s3.us-west-2.amazonaws.com/evaluation/{context_name}/data/df_results_w_norm_err.parquet"  # load from s3 (public)
    eval_save_path = f"{context_name}/output"

    df_results = load_pd.load(path=df_result_save_path_w_norm_err)

    paper_run = PaperRunTabArena(repo=None, output_dir=eval_save_path)
    paper_run.eval(df_results=df_results)
