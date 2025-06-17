from __future__ import annotations

import shutil

from tabrepo.nips2025_utils.tabarena_context import TabArenaContext


if __name__ == '__main__':
    download_results: bool | str = "auto"  # results must be downloaded for the script to work
    elo_bootstrap_rounds = 100  # 1 for toy, 100 for paper
    save_path = "output_paper_results"  # folder to save all figures and tables

    tabarena_context = TabArenaContext()
    df_results = tabarena_context.load_results_paper(download_results=download_results)
    tabarena_context.evaluate_all(df_results=df_results, save_path=save_path, elo_bootstrap_rounds=elo_bootstrap_rounds)

    zip_results = True
    upload_to_s3 = False
    if zip_results:
        file_prefix = f"tabarena51_paper_results"
        file_name = f"{file_prefix}.zip"
        shutil.make_archive(file_prefix, 'zip', root_dir=save_path)
        if upload_to_s3:
            from autogluon.common.utils.s3_utils import upload_file
            upload_file(file_name=file_name, bucket="tabarena", prefix=save_path)
