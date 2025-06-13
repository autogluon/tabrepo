from __future__ import annotations

from pathlib import Path

import pandas as pd

from tabrepo.nips2025_utils.fetch_metadata import load_task_metadata
from tabrepo.paper.tabarena_evaluator import TabArenaEvaluator


def _get_problem_type(n_classes: int):
    if n_classes < 2:
        return "regression"
    elif n_classes == 2:
        return "binary"
    else:
        return "multiclass"


def evaluate_all(df_results: pd.DataFrame, eval_save_path: str | Path, elo_bootstrap_rounds: int = 100):
    datasets_tabpfn = list(load_task_metadata(subset="TabPFNv2")["name"])
    datasets_tabicl = list(load_task_metadata(subset="TabICL")["name"])
    task_metadata = load_task_metadata()

    task_metadata["problem_type"] = task_metadata["NumberOfClasses"].apply(_get_problem_type)

    eval_save_path = Path(eval_save_path)

    # plots for sub-benchmarks, with and without imputation
    for use_tabpfn in [False, True]:
        for use_tabicl in [False, True]:
            for use_imputation in [False, True]:
                for problem_type in [None, "cls", "reg"]:
                    for lite in [False, True]:
                        folder_name = ("tabpfn-tabicl" if use_tabpfn else "tabicl") \
                            if use_tabicl else ("tabpfn" if use_tabpfn else "full")
                        if lite:
                            folder_name = str(Path("lite") / folder_name)
                        if use_imputation:
                            folder_name = folder_name + "-imputed"
                        if problem_type is not None:
                            folder_name = folder_name + f"-{problem_type}"

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
                        if datasets is None:
                            datasets = list(task_metadata["name"])

                        if problem_type is None:
                            problem_types = None
                        elif problem_type == "cls":
                            problem_types = ["binary", "multiclass"]
                        elif problem_type == "reg":
                            problem_types = ["regression"]
                        else:
                            raise AssertionError(f"Invalid problem_type value: {problem_type}")

                        if problem_type == "reg":
                            banned_model_types.append("TABICL")
                            banned_model_types = list(set(banned_model_types))

                        if problem_types:
                            datasets = [d for d in datasets if task_metadata[task_metadata["name"] == d].iloc[0]["problem_type"] in problem_types]

                        if len(datasets) == 0:
                            continue

                        plotter = TabArenaEvaluator(
                            output_dir=eval_save_path / folder_name,
                            datasets=datasets,
                            problem_types=problem_types,
                            banned_model_types=None if use_imputation else banned_model_types,
                            elo_bootstrap_rounds=elo_bootstrap_rounds,
                            folds=[0] if lite else None,
                        )

                        plotter.eval(
                            df_results=df_results,
                            imputed_names=imputed_models,
                            only_datasets_for_method={'TabPFNv2': datasets_tabpfn, 'TabICL': datasets_tabicl},
                            plot_extra_barplots=True,
                            plot_times=True,
                            plot_other=False,
                        )
