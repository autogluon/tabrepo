from __future__ import annotations

import shutil

from tabrepo.nips2025_utils.tabarena_context import TabArenaContext
from tabrepo.nips2025_utils.artifacts import tabarena_method_metadata_collection


if __name__ == '__main__':
    download_results: bool | str = "auto"  # results must be downloaded for the script to work
    elo_bootstrap_rounds = 100  # 1 for toy, 100 for paper
    save_path = "output_paper_results"  # folder to save all figures and tables
    use_latex: bool = False  # Set to True if you have the appropriate latex packages installed for nicer figure style

    include_2025_09_03_results = True  # Set to True to include new results not in the paper preprint

    # TODO: This is old, regenerate portfolio with new results (such as RealMLP_GPU)
    extra_methods = [
        "Portfolio-N200-4h",
    ]

    extra_methods = [tabarena_method_metadata_collection.get_method_metadata(method=m) for m in extra_methods]

    if include_2025_09_03_results:
        tabarena_context = TabArenaContext(
            extra_methods=extra_methods,
            include_ag_140=True,
            include_mitra=True,
        )
        df_results_holdout = None  # TODO: Mitra does not yet have holdout results saved in S3, need to add
    else:
        tabarena_context = TabArenaContext(
            extra_methods=extra_methods,
            include_ag_140=False,
            include_mitra=False,
        )
        df_results_holdout = tabarena_context.load_results_paper(download_results=download_results, holdout=True)
    df_results = tabarena_context.load_results_paper(download_results=download_results)

    if include_2025_09_03_results:
        fillna_method = "RF (default)"
        df_results = TabArenaContext.fillna_metrics(
            df_to_fill=df_results,
            df_fillna=df_results[df_results["method"] == fillna_method],
        )

    cpu_methods = [
        "ModernNCA",
        # TODO: Remove RealMLP CPU since new Sept GPU ver shouldn't be compared to CPU run in June.
        "RealMLP",
        "TabM",
    ]
    extra_methods_cpu = [tabarena_method_metadata_collection.get_method_metadata(method=m) for m in cpu_methods]

    tabarena_context_cpu = TabArenaContext(methods=extra_methods_cpu)
    df_results_cpu = tabarena_context_cpu.load_results_paper(methods=cpu_methods, download_results=download_results)

    tabarena_context = TabArenaContext(
        methods=tabarena_context.method_metadata_collection.method_metadata_lst + tabarena_context_cpu.method_metadata_collection.method_metadata_lst
    )

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
