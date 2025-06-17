from __future__ import annotations

import shutil

from tabrepo.nips2025_utils.artifacts.tabarena51_artifact_loader import TabArena51ArtifactLoader
from tabrepo.nips2025_utils.eval_all import evaluate_all
from tabrepo.nips2025_utils.tabarena_context import TabArenaContext


if __name__ == '__main__':
    download_results: bool | str = "auto"  # results must be downloaded for the script to work
    elo_bootstrap_rounds = 100  # 3 for toy, 100 for paper

    zip_results = True
    upload_to_s3 = False
    eval_save_path = "output_paper_results"
    file_prefix = f"tabarena51_paper_results"
    file_name = f"{file_prefix}.zip"

    loader = TabArena51ArtifactLoader()
    if isinstance(download_results, bool) and download_results:
        loader.download_results()

    tabarena_context = TabArenaContext()
    try:
        df_results = tabarena_context.load_results_paper()
    except FileNotFoundError as err:
        if isinstance(download_results, str) and download_results == "auto":
            print(f"Missing local results files! Attempting to download them and retry...")
            loader.download_results()
            df_results = tabarena_context.load_results_paper()
        else:
            print(f"Missing local results files! Try setting `download_results=True` to get the required files.")
            raise err

    evaluate_all(df_results=df_results, eval_save_path=eval_save_path, elo_bootstrap_rounds=elo_bootstrap_rounds)

    if zip_results:
        shutil.make_archive(file_prefix, 'zip', root_dir=eval_save_path)
        if upload_to_s3:
            from autogluon.common.utils.s3_utils import upload_file
            upload_file(file_name=file_name, bucket="tabarena", prefix=eval_save_path)
