import pandas as pd

from autogluon.common.loaders import load_pd


def load_results(
    results: str,
    results_by_dataset: str,
    raw: str,
    metadata: str,
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    df_results = load_pd.load(results)
    df_results_by_dataset = load_pd.load(results_by_dataset)
    df_metadata = load_pd.load(metadata)
    df_raw = load_pd.load(raw)
    df_raw['tid_new'] = df_raw['tid'].astype(int).astype(str) + '_' + df_raw['fold'].astype(str)
    return df_results, df_results_by_dataset, df_raw, df_metadata


def combine_results_with_score_val(df_raw, df_results_by_dataset):
    df_raw_zoom = df_raw[['model', 'tid_new', 'score_val', 'fold', 'tid']].copy()
    df_raw_zoom['dataset'] = df_raw_zoom['tid_new']
    df_raw_zoom['framework'] = df_raw_zoom['model']
    df_raw_zoom = df_raw_zoom[['framework', 'dataset', 'score_val', 'fold', 'tid']]
    df_results_by_dataset_with_score_val = df_results_by_dataset.merge(df_raw_zoom, on=['framework', 'dataset'])
    return df_results_by_dataset_with_score_val
