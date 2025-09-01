from pathlib import Path

import pandas as pd

from tabrepo.nips2025_utils.end_to_end import EndToEnd, EndToEndResults


"""
First refer to `run_jobs_lightgbm_demo.py

# Get required input files
aws s3 cp --recursive "s3://prateek-ag/tabarena-lightgbm-demo" ../data/tabarena-lightgbm-demo/ --exclude "*.log"
"""
if __name__ == '__main__':
    method = "LightGBM_demo"
    name_suffix = "_demo"
    path_raw = Path(
        "/home/ubuntu/workspace/data/tabarena-lightgbm-demo/data/"
    )
    fig_output_dir = Path("tabarena_figs") / method
    cache = True

    """
    Run logic end-to-end and cache all results:
    1. load raw artifacts
        path_raw should be a directory containing `results.pkl` files for each run.
        In the current code, we require `path_raw` to contain the results of only 1 type of method.
    2. infer method_metadata
    3. cache method_metadata
    4. cache raw artifacts
    5. infer task_metadata
    5. generate processsed
    6. cache processed
    7. generate results
    8. cache results

    Once this is executed once, it does not need to be ran again.
    """
    if cache:
        end_to_end = EndToEnd.from_path_raw(path_raw=path_raw, name_suffix=name_suffix)

    """
    Load cached results and compare on TabArena
    1. Generates figures and leaderboard using the TabArena methods and the user's method
    2. Compares on all datasets if `filter_dataset_fold=False`, else only tasks from the user's method if `filter_dataset_fold=True`.
    3. Missing values are imputed to default RandomForest.
    """
    end_to_end_results = EndToEndResults.from_cache(method=method)

    leaderboard: pd.DataFrame = end_to_end_results.compare_on_tabarena(
        output_dir=fig_output_dir,
        filter_dataset_fold=True,
    )

    print(leaderboard.to_markdown())
