from __future__ import annotations

import shutil
from pathlib import Path

from tabarena.nips2025_utils.tabarena_context import TabArenaContext
from tabarena.nips2025_utils.artifacts import tabarena_method_metadata_collection
from tabarena.nips2025_utils.artifacts._tabarena_method_metadata import (
    tabarena_method_metadata_2025_06_12_collection_main,
    tabarena_method_metadata_2025_06_12_collection_gpu_ablation,
)

from advanced.rebuttal.run_plot_pareto_over_tuning_time import plot_pareto_n_configs


if __name__ == '__main__':
    download_results: bool | str = "auto"  # results must be downloaded for the script to work
    elo_bootstrap_rounds = 200  # 1 for toy, 200 for paper
    save_path = "output_paper_results"  # folder to save all figures and tables
    use_latex: bool = False  # Set to True if you have the appropriate latex packages installed for nicer figure style

    include_2025_09_03_results = True  # Set to True to include new results not in the paper preprint
    only_2025_06_12_results = True
    plot_n_configs = False

    if only_2025_06_12_results:
        save_path = str(Path(save_path) / "2025_06_12")

        tabarena_context = TabArenaContext(
            methods=tabarena_method_metadata_2025_06_12_collection_main.method_metadata_lst,
            include_ag_140=False,
            include_mitra=False,
        )
        df_results_holdout = tabarena_context.load_results_paper(download_results=download_results, holdout=True)
    else:
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

    if only_2025_06_12_results:
        cpu_methods = [
            "ModernNCA",
            # TODO: Remove RealMLP CPU since new Sept GPU ver shouldn't be compared to CPU run in June.
            "RealMLP_GPU",
            "TabM",
        ]
        extra_methods_cpu = tabarena_method_metadata_2025_06_12_collection_gpu_ablation.method_metadata_lst
    else:
        cpu_methods = [
            "ModernNCA",
            # TODO: Remove RealMLP CPU since new Sept GPU ver shouldn't be compared to CPU run in June.
            "RealMLP",
            "TabM",
        ]
        extra_methods_cpu = [tabarena_method_metadata_collection.get_method_metadata(method=m) for m in cpu_methods]
    tabarena_context_cpu = TabArenaContext(methods=extra_methods_cpu, include_mitra=False, include_ag_140=False)
    df_results_cpu = tabarena_context_cpu.load_results_paper(methods=cpu_methods, download_results=download_results)

    configs_hyperparameters = tabarena_context.load_configs_hyperparameters(download=download_results)

    tabarena_context_all = TabArenaContext(
        methods=tabarena_context.method_metadata_collection.method_metadata_lst + tabarena_context_cpu.method_metadata_collection.method_metadata_lst
    )

    if plot_n_configs:
        plot_pareto_n_configs(
            fig_save_dir=Path(save_path) / "n_configs",
            average_seeds=True,
        )

        plot_pareto_n_configs(
            fig_save_dir=Path(save_path) / "no_average_seeds" / "n_configs",
            average_seeds=False,
        )

    tabarena_context_all.evaluate_all(
        df_results=df_results,
        df_results_holdout=df_results_holdout,
        df_results_cpu=df_results_cpu,
        configs_hyperparameters=configs_hyperparameters,
        save_path=save_path,
        elo_bootstrap_rounds=elo_bootstrap_rounds,
        use_latex=use_latex,
        realmlp_cpu=only_2025_06_12_results,
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
