import pandas as pd


# FIXME: Doesn't work for multi-fold
def get_dataset_to_tid_dict(df_raw: pd.DataFrame) -> dict:
    df_tid_to_dataset_map = df_raw[['tid', 'dataset']].drop_duplicates(['tid', 'dataset'])
    dataset_to_tid_dict = df_tid_to_dataset_map.set_index('dataset')
    dataset_to_tid_dict = dataset_to_tid_dict['tid'].to_dict()
    return dataset_to_tid_dict


def get_dataset_name_to_tid_dict(df_raw: pd.DataFrame) -> dict:
    df_tid_to_dataset_map = df_raw[['task', 'tid']].drop_duplicates(['task', 'tid'])
    dataset_to_tid_dict = df_tid_to_dataset_map.set_index('task')
    # dataset_to_tid_dict = dataset_to_tid_dict[dataset_to_tid_dict['fold'] == 0]
    dataset_to_tid_dict = dataset_to_tid_dict['tid'].to_dict()
    return dataset_to_tid_dict


def filter_datasets(df_results_by_dataset: pd.DataFrame, df_raw: pd.DataFrame, datasets: pd.DataFrame):
    df_results_by_dataset = df_results_by_dataset.merge(datasets, on=["dataset", "fold"])
    df_raw = df_raw.merge(datasets, on=["dataset", "fold"])
    return df_results_by_dataset, df_raw


def get_dataset_to_metric_problem_type(df_raw: pd.DataFrame) -> pd.DataFrame:
    df_raw_min = df_raw[["dataset", "metric", "problem_type"]].drop_duplicates()
    counts = df_raw_min["dataset"].value_counts().to_dict()
    for dataset in counts:
        if counts[dataset] != 1:
            df_raw_dataset = df_raw_min[df_raw_min["dataset"] == dataset]
            raise AssertionError(f"Error: Multiple `problem_type` or `metric` values defined in the data for dataset {dataset}\n:"
                                 f"{df_raw_dataset}")
    df_raw_min = df_raw_min.set_index("dataset")
    return df_raw_min
