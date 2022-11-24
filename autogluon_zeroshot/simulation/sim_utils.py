import pandas as pd


# FIXME: Doesn't work for multi-fold
def get_dataset_to_tid_dict(df_raw: pd.DataFrame) -> dict:
    df_tid_to_dataset_map = df_raw[['tid', 'dataset']].drop_duplicates(['tid', 'dataset'])
    dataset_to_tid_dict = df_tid_to_dataset_map.set_index('dataset')
    dataset_to_tid_dict = dataset_to_tid_dict['tid'].to_dict()
    return dataset_to_tid_dict


def get_dataset_name_to_tid_dict(df_raw: pd.DataFrame) -> dict:
    df_tid_to_dataset_map = df_raw[['tid_new', 'tid']].drop_duplicates(['tid_new', 'tid'])
    dataset_to_tid_dict = df_tid_to_dataset_map.set_index('tid_new')
    # dataset_to_tid_dict = dataset_to_tid_dict[dataset_to_tid_dict['fold'] == 0]
    dataset_to_tid_dict = dataset_to_tid_dict['tid'].to_dict()
    return dataset_to_tid_dict



def filter_datasets(df_results_by_dataset, df_raw, datasets: set):
    df_results_by_dataset = df_results_by_dataset[df_results_by_dataset['dataset'].isin(datasets)]
    df_raw = df_raw[df_raw['tid_new'].isin(datasets)]
    return df_results_by_dataset, df_raw
