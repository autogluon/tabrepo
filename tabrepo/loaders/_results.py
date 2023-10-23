import pandas as pd

from autogluon.common.loaders import load_pd


def load_results(
    results_by_dataset: str,
    raw: str,
    metadata: str,
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
        # FIXME: HACK
        if "fold" not in df_results_by_dataset.columns:
            from autogluon.bench.eval.evaluation.evaluate_utils import convert_to_dataset_fold
            df_results_by_dataset = convert_to_dataset_fold(df_results_by_dataset, column_name="tid", column_type=int)
            # df_results_by_dataset["dataset"] = df_results_by_dataset["dataset"].astype(int)
    else:
        df_results_by_dataset = df_raw[[
            "framework",
            "tid",
            "fold",
            "metric_error",
            "time_train_s",
            "time_infer_s",
        ]].copy(deep=True)

    if require_tid_in_metadata:
        if df_metadata is None:
            raise AssertionError('`metadata` must not be None if `require_tid_in_metadata=True`')
        valid_tids = set(list(df_metadata['tid'].unique()))
        init_tid_count = len(df_raw['tid'].unique())
        df_raw = df_raw[df_raw['tid'].isin(valid_tids)]
        post_tid_count = len(df_raw['tid'].unique())
        if init_tid_count != post_tid_count:
            print(f'Filtered datasets with tids that are missing from metadata. '
                  f'Filtered from {init_tid_count} datasets -> {post_tid_count} datasets.')

    return df_results_by_dataset, df_raw, df_metadata


def get_metric_name(metric: str) -> str:
    metric_map = {
        "auc": "roc_auc",
        "neg_rmse": "rmse",
        "neg_logloss": "log_loss",
    }
    return metric_map.get(metric, metric)


def get_metric_from_raw(row):
    from autogluon.core.metrics import get_metric
    metric = row["metric"]
    score_val = row["score_val"]

    metric = get_metric_name(metric=metric)
    return get_metric(metric=metric).convert_score_to_error(score=score_val)


def preprocess_raw(df_raw: pd.DataFrame, inplace=True) -> pd.DataFrame:
    if not inplace:
        df_raw = df_raw.copy(deep=True)
    df_raw['tid'] = df_raw['tid'].astype(int)
    if "metric_error_val" not in df_raw:
        df_raw["metric"] = df_raw["metric"].apply(lambda m: get_metric_name(metric=m))
        df_raw["metric_error_val"] = df_raw[["score_val", "metric"]].apply(
            lambda row: get_metric_from_raw(row=row), axis=1,
        )
    if "model" in df_raw:
        df_raw['framework'] = df_raw['model']
    return df_raw


def preprocess_comparison(df_comparison_raw: pd.DataFrame, inplace=True) -> pd.DataFrame:
    if not inplace:
        df_comparison_raw = df_comparison_raw.copy(deep=True)

    # FIXME: HACK
    if "fold" not in df_comparison_raw.columns:
        from autogluon.bench.eval.evaluation.evaluate_utils import convert_to_dataset_fold
        df_comparison_raw = convert_to_dataset_fold(df_comparison_raw, column_name="tid", column_type=int)

    df_comparison_raw['tid'] = df_comparison_raw['tid'].astype(int)
    return df_comparison_raw


def combine_results_with_score_val(df_raw, df_results_by_dataset):
    df_raw_zoom = df_raw[['framework', 'metric_error_val', 'fold', 'tid']].copy()
    df_raw_zoom = df_raw_zoom[['framework', 'metric_error_val', 'fold', 'tid']]
    df_results_by_dataset_with_score_val = df_results_by_dataset.merge(df_raw_zoom, on=['framework', 'tid', 'fold'])
    return df_results_by_dataset_with_score_val
