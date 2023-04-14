import pandas as pd

from autogluon.common.loaders import load_pd


def load_results(
    results: str,
    results_by_dataset: str,
    raw: str,
    metadata: str,
    require_tid_in_metadata: bool = False,
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    print(f'Loading input files...\n'
          f'\tresults:            {results}\n'
          f'\tresults_by_dataset: {results_by_dataset}\n'
          f'\traw:                {raw}\n'
          f'\tmetadata:           {metadata}')
    df_results = load_pd.load(results)
    df_results_by_dataset = load_pd.load(results_by_dataset)
    if metadata is not None:
        df_metadata = load_pd.load(metadata)
    else:
        df_metadata = None
    df_raw = load_pd.load(raw)
    df_raw['tid'] = df_raw['tid'].astype(int)
    df_raw['tid_new'] = df_raw['tid'].astype(str) + '_' + df_raw['fold'].astype(str)

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

    return df_results, df_results_by_dataset, df_raw, df_metadata


def combine_results_with_score_val(df_raw, df_results_by_dataset):
    df_raw_zoom = df_raw[['model', 'tid_new', 'score_val', 'fold', 'tid']].copy()
    df_raw_zoom['dataset'] = df_raw_zoom['tid_new']
    df_raw_zoom['framework'] = df_raw_zoom['model']
    df_raw_zoom = df_raw_zoom[['framework', 'dataset', 'score_val', 'fold', 'tid']]
    df_results_by_dataset_with_score_val = df_results_by_dataset.merge(df_raw_zoom, on=['framework', 'dataset'])
    return df_results_by_dataset_with_score_val
