from __future__ import annotations

from pathlib import Path

import pandas as pd

from tabrepo.nips2025_utils.fetch_metadata import load_task_metadata
from tabrepo.paper.tabarena_evaluator import TabArenaEvaluator


def evaluate_all(df_results: pd.DataFrame, eval_save_path: str | Path, elo_bootstrap_rounds: int = 100):
    datasets_tabpfn = list(load_task_metadata(subset="TabPFNv2")["name"])
    datasets_tabicl = list(load_task_metadata(subset="TabICL")["name"])

    # plots for sub-benchmarks, with and without imputation
    for use_tabpfn in [False, True]:
        for use_tabicl in [False, True]:
            for use_imputation in [False, True]:
                for lite in [False, True]:
                    if use_imputation and lite:
                        continue

                    folder_name = ("tabpfn-tabicl" if use_tabpfn else "tabicl") \
                        if use_tabicl else ("tabpfn" if use_tabpfn else "full")
                    if lite:
                        folder_name = folder_name + "-lite"
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

                    datasets = (
                        list(set(datasets_tabpfn).intersection(datasets_tabicl)) if use_tabpfn else datasets_tabicl) \
                        if use_tabicl else (datasets_tabpfn if use_tabpfn else None)

                    plotter = TabArenaEvaluator(
                        output_dir=eval_save_path / folder_name,
                        datasets=datasets,
                        banned_model_types=None if use_imputation else banned_model_types,
                        elo_bootstrap_rounds=elo_bootstrap_rounds,
                        folds=[0] if lite else None,
                    )

                    plotter.eval(df_results=df_results, imputed_names=imputed_models,
                                 only_datasets_for_method={'TabPFNv2': datasets_tabpfn, 'TabICL': datasets_tabicl},
                                 plot_extra_barplots='full' in folder_name, plot_times='full' in folder_name,
                                 plot_other=False)
