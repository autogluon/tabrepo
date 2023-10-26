import pandas as pd

from autogluon.common.loaders import load_pd


def load_results(
    results_by_dataset: str,
    raw: str,
    metadata: str,
    metadata_join_column: str = "dataset",
    require_tid_in_metadata: bool = False,
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    print(f'Loading input files...\n'
          f'\tresults_by_dataset: {results_by_dataset}\n'
          f'\traw:                {raw}\n'
          f'\tmetadata:           {metadata}')
    if metadata is not None:
        df_metadata = load_pd.load(metadata)
    else:
        df_metadata = None
    df_raw = load_pd.load(raw)
    df_raw = preprocess_raw(df_raw=df_raw, inplace=True)

    if results_by_dataset is not None:
        df_results_by_dataset = load_pd.load(results_by_dataset)
    else:
        df_results_by_dataset = df_raw[[
            "framework",
            "dataset",
            "fold",
            "metric_error",
            "time_train_s",
            "time_infer_s",
        ]].copy(deep=True)

    if require_tid_in_metadata:
        if df_metadata is None:
            raise AssertionError('`metadata` must not be None if `require_tid_in_metadata=True`')
        if metadata_join_column not in df_raw:
            raise AssertionError(f"`metadata` file was specified but `raw` is missing the required `{metadata_join_column}` column to join with `metadata`.\n"
                                 f"\tmetadata_join_column: {metadata_join_column}\n"
                                 f"\tpath             raw: {raw}\n"
                                 f"\tpath        metadata: {metadata}\n"
                                 f"\tcolumns          raw: {list(df_raw.columns)}\n"
                                 f"\tcolumns     metadata: {list(df_metadata.columns)}")
        if metadata_join_column not in df_metadata:
            raise AssertionError(f"`metadata` file was specified but `metadata` is missing the required `{metadata_join_column}` column to join with `raw`.\n"
                                 f"\tmetadata_join_column: {metadata_join_column}\n"
                                 f"\tpath             raw: {raw}\n"
                                 f"\tpath        metadata: {metadata}\n"
                                 f"\tcolumns          raw: {list(df_raw.columns)}\n"
                                 f"\tcolumns     metadata: {list(df_metadata.columns)}")
        valid_values = set(list(df_metadata[metadata_join_column].unique()))
        init_dataset_count = len(df_raw["dataset"].unique())
        df_raw = df_raw[df_raw[metadata_join_column].isin(valid_values)]
        post_dataset_count = len(df_raw["dataset"].unique())
        if init_dataset_count != post_dataset_count:
            print(f'Filtered datasets with tids that are missing from metadata. '
                  f'Filtered from {init_dataset_count} datasets -> {post_dataset_count} datasets.')

    if df_metadata is not None:
        if "dataset" != metadata_join_column:
            unique_metadata_join_column_vals = df_raw[[metadata_join_column, "dataset"]].drop_duplicates()
            assert unique_metadata_join_column_vals["dataset"].value_counts().values.max() == 1, (f"Metadata join key `{metadata_join_column}` does not produce a 1:1 mapping with `dataset`.\n"
                                                                                                  f"Counts post-join (should always be 1):\n"
                                                                                                  f"{unique_metadata_join_column_vals['dataset'].value_counts()}")
            df_metadata = df_metadata.drop(columns=["dataset"], errors="ignore")
        else:
            unique_metadata_join_column_vals = df_raw[[metadata_join_column]]
        df_metadata = df_metadata.merge(unique_metadata_join_column_vals, on=[metadata_join_column])
        metadata_column_order = ["dataset"] + [c for c in df_metadata.columns if c != "dataset"]
        df_metadata = df_metadata[metadata_column_order]  # make dataset first

    return df_results_by_dataset, df_raw, df_metadata


def get_metric_name(metric: str) -> str:
    metric_map = {
        "auc": "roc_auc",
        "neg_rmse": "rmse",
        "neg_logloss": "log_loss",
    }
    return metric_map.get(metric, metric)


def get_metric_error_from_score(score: float, metric: str) -> float:
    from autogluon.core.metrics import get_metric
    metric = get_metric_name(metric=metric)
    return get_metric(metric=metric).convert_score_to_error(score=score)


def preprocess_raw(df_raw: pd.DataFrame, inplace=True) -> pd.DataFrame:
    if not inplace:
        df_raw = df_raw.copy(deep=True)
    if "tid" in df_raw:
        df_raw['tid'] = df_raw['tid'].astype(int)
    if "metric_error_val" not in df_raw:
        df_raw["metric"] = df_raw["metric"].apply(lambda m: get_metric_name(metric=m))
        df_raw["metric_error_val"] = df_raw[["score_val", "metric"]].apply(
            lambda row: get_metric_error_from_score(score=row["score_val"], metric=row["metric"]), axis=1,
        )
    if "model" in df_raw:
        df_raw['framework'] = df_raw['model']
    return df_raw


def preprocess_comparison(df_comparison_raw: pd.DataFrame, inplace=True) -> pd.DataFrame:
    if not inplace:
        df_comparison_raw = df_comparison_raw.copy(deep=True)
    return df_comparison_raw


def combine_results_with_score_val(df_raw: pd.DataFrame, df_results_by_dataset: pd.DataFrame) -> pd.DataFrame:
    df_raw_zoom = df_raw[['framework', 'metric_error_val', 'fold', 'dataset']].copy()
    df_raw_zoom = df_raw_zoom[['framework', 'metric_error_val', 'fold', 'dataset']]
    df_results_by_dataset_with_score_val = df_results_by_dataset.merge(df_raw_zoom, on=['framework', 'dataset', 'fold'])
    return df_results_by_dataset_with_score_val
