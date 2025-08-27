from __future__ import annotations

from pathlib import Path

import pandas as pd

from tabrepo.nips2025_utils.tabarena_context import TabArenaContext
from tabrepo.paper.tabarena_evaluator import TabArenaEvaluator


def compare_on_tabarena(
    output_dir: str | Path,
    df_metrics: pd.DataFrame,
    *,
    filter_dataset_fold: bool = False,
    df_results_extra: pd.DataFrame = None,
    subset: str | list[str] | None = None,
) -> pd.DataFrame:
    df_metrics = df_metrics.copy(deep=True)
    if df_results_extra is not None:
        df_results_extra = df_results_extra.copy(deep=True)

    output_dir = Path(output_dir)

    tabarena_context = TabArenaContext()

    fillna_method = "RF (default)"
    paper_results = tabarena_context.load_results_paper(
        download_results="auto",
        methods_drop=["Portfolio-N200-4h"],  # TODO: Clean this up by not including by default
    )

    if filter_dataset_fold:
        paper_results = filter_to_valid_tasks(
            df_to_filter=paper_results,
            df_filter=df_metrics,
        )
        if df_results_extra is not None:
            df_results_extra = filter_to_valid_tasks(
                df_to_filter=df_results_extra,
                df_filter=df_metrics,
            )

    # FIXME: Nick: After imputing: ta_name, ta_suite, config_type, etc. are incorrect,
    #  need to use original, not filled values
    #  This doesn't impact the evaluation, but could introduce bugs in future if we use these columns
    #  Fixing this is do-able, but requires some complex pandas tricks, so I haven't had time to implement it yet
    df_metrics = TabArenaContext.fillna_metrics(
        df_metrics=df_metrics,
        df_fillna=paper_results[paper_results["method"] == fillna_method],
    )

    df_results = pd.concat([paper_results, df_metrics], ignore_index=True)

    if df_results_extra is not None:
        if filter_dataset_fold:
            df_results = filter_to_valid_tasks(
                df_to_filter=df_results,
                df_filter=df_results_extra,
            )

        df_results_extra = TabArenaContext.fillna_metrics(
            df_metrics=df_results_extra,
            df_fillna=df_results[df_results["method"] == fillna_method],
        )

        df_results = pd.concat([df_results, df_results_extra], ignore_index=True)

    if subset is not None:
        if isinstance(subset, str):
            subset = [subset]
        df_results = subset_tasks(df_results=df_results, subset=subset)

    # Handle imputation of names
    imputed_names = list(df_results["method"][df_results["imputed"] > 0].unique())
    if len(imputed_names) == 0:
        imputed_names = None
    if imputed_names is not None:
        from tabrepo.paper.paper_utils import get_method_rename_map

        # remove suffix
        imputed_names = [n.split(" (")[0] for n in imputed_names]
        imputed_names = [get_method_rename_map().get(n, n) for n in imputed_names]
        imputed_names = list(set(imputed_names))
        if "KNN" in imputed_names:
            imputed_names.remove("KNN")
        print(f"Model for which results were imputed: {imputed_names}")

    baselines = list(
        df_results[
            df_results["method_type"].isin(['baseline', 'portfolio']) |
            ((df_results["method_type"] == "config") & df_results["method_subtype"].isna())
        ]["method"].unique()
    )

    plotter = TabArenaEvaluator(
        output_dir=output_dir,
    )
    return plotter.eval(
        df_results=df_results,
        baselines=baselines,
        plot_extra_barplots=False,
        plot_times=True,
        plot_other=False,
        imputed_names=imputed_names,
    )


def filter_to_valid_tasks(df_to_filter: pd.DataFrame, df_filter: pd.DataFrame) -> pd.DataFrame:
    dataset_fold_map = df_filter.groupby("dataset")["fold"].apply(set)

    def is_in(dataset: str, fold: int) -> bool:
        return (dataset in dataset_fold_map.index) and (fold in dataset_fold_map.loc[dataset])

    # filter `df_to_filter` to only the dataset, fold pairs that are present in `df_filter`
    is_in_lst = [
        is_in(dataset, fold) for dataset, fold in zip(
            df_to_filter["dataset"],
            df_to_filter["fold"],
        )]
    df_filtered = df_to_filter[is_in_lst]
    return df_filtered


def subset_tasks(df_results: pd.DataFrame, subset: list[str]) -> pd.DataFrame:
    from tabrepo.nips2025_utils.fetch_metadata import load_task_metadata

    df_results = df_results.copy(deep=True)
    for filter_subset in subset:
        if filter_subset == "classification":
            df_results = df_results[
                df_results["problem_type"].isin(["binary", "multiclass"])
            ]
        elif filter_subset == "regression":
            df_results = df_results[df_results["problem_type"] == "regression"]
        elif filter_subset == "medium+":
            task_metadata = load_task_metadata()
            task_metadata = task_metadata[task_metadata["n_samples_train_per_fold"] >= 10000]
            valid_datasets = task_metadata["dataset"].unique()
            df_results = df_results[df_results["dataset"].isin(valid_datasets)]
        elif filter_subset == "small":
            task_metadata = load_task_metadata()
            task_metadata = task_metadata[task_metadata["n_samples_train_per_fold"] < 10000]
            valid_datasets = task_metadata["dataset"].unique()
            df_results = df_results[df_results["dataset"].isin(valid_datasets)]
        elif filter_subset == "lite":
            df_results = df_results[df_results["fold"] == 0]
        elif filter_subset == "tabicl":
            allowed_dataset = load_task_metadata(subset="TabICL")[
                "dataset"
            ].tolist()
            df_results = df_results[df_results["dataset"].isin(allowed_dataset)]
        elif filter_subset == "tabpfn":
            allowed_dataset = load_task_metadata(subset="TabPFNv2")[
                "dataset"
            ].tolist()
            df_results = df_results[df_results["dataset"].isin(allowed_dataset)]
        elif filter_subset == "tabpfn/tabicl":
            ad_tabicl = load_task_metadata(subset="TabICL")["dataset"].tolist()
            ad_tabpfn = load_task_metadata(subset="TabPFNv2")["dataset"].tolist()
            allowed_dataset = list(set(ad_tabicl).intersection(set(ad_tabpfn)))
            df_results = df_results[df_results["dataset"].isin(allowed_dataset)]
        else:
            raise ValueError(f"Invalid subset {subset} name!")
        df_results = df_results.reset_index(drop=True)
    return df_results
