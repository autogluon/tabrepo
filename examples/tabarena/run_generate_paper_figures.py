from __future__ import annotations

import shutil

from tabrepo.nips2025_utils.tabarena_context import TabArenaContext


if __name__ == '__main__':
    download_results: bool | str = "auto"  # results must be downloaded for the script to work
    elo_bootstrap_rounds = 100  # 1 for toy, 100 for paper
    save_path = "output_paper_results"  # folder to save all figures and tables
    use_latex: bool = False  # Set to True if you have the appropriate latex packages installed for nicer figure style

    tabarena_context = TabArenaContext()
    df_results = tabarena_context.load_results_paper(download_results=download_results)
    df_results_holdout = tabarena_context.load_results_paper(download_results=download_results, holdout=True)

    cpu_methods = [
        "ModernNCA",
        "RealMLP",
        "TabM",
    ]
    df_results_cpu = tabarena_context.load_results_paper(methods=cpu_methods, download_results=download_results)

    configs_hyperparameters = tabarena_context.load_configs_hyperparameters(download=download_results)

    tabarena_context.evaluate_all(
        df_results=df_results,
        df_results_holdout=df_results_holdout,
        df_results_cpu=df_results_cpu,
        configs_hyperparameters=configs_hyperparameters,
        save_path=save_path,
        elo_bootstrap_rounds=elo_bootstrap_rounds,
        use_latex=use_latex,
    )

    zip_results = True
    upload_to_s3 = False
    if zip_results:
        file_prefix = f"tabarena51_paper_results"
        file_name = f"{file_prefix}.zip"
        shutil.make_archive(file_prefix, 'zip', root_dir=save_path)
        if upload_to_s3:
            from autogluon.common.utils.s3_utils import upload_file
            upload_file(file_name=file_name, bucket="tabarena", prefix=save_path)
