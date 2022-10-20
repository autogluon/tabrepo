
from autogluon_zeroshot.utils.result_utils import shrink_result_file_size, shrink_ranked_result_file_size


if __name__ == '__main__':
    # Compresses CSV to Parquet (311.2 MB -> 19.2 MB)
    path_load = '../data/results/all_v3/openml_ag_2022_10_13_zs_models.csv'
    path_save = '../data/results/all_v3/openml_ag_2022_10_13_zs_models.parquet'
    shrink_result_file_size(path_load=path_load, path_save=path_save)

    # Compresses CSV to Parquet (76.4 MB -> 22 MB)
    path_load = '../data/results/all_v3/results_ranked_by_dataset_valid.csv'
    path_save = '../data/results/all_v3/results_ranked_by_dataset_valid.parquet'
    shrink_ranked_result_file_size(path_load=path_load, path_save=path_save)
