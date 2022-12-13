from pathlib import Path

from autogluon_zeroshot.utils.result_utils import shrink_result_file_size, shrink_ranked_result_file_size


if __name__ == '__main__':
    result_root = Path(__file__).parent.parent / 'data' / 'results' / 'bagged'

    # Compresses CSV to Parquet (311.2 MB -> 19.2 MB)
    path_load = str(result_root / 'openml_ag_2022_12_11_zs_models.csv')
    path_save = str(result_root / 'openml_ag_2022_12_11_zs_models.parquet')
    shrink_result_file_size(path_load=path_load, path_save=path_save)

    result_root = Path(__file__).parent.parent / 'data' / 'results' / 'bagged' / 'small_30'
    # Compresses CSV to Parquet (76.4 MB -> 22 MB)
    path_load = str(result_root / 'results_ranked_by_dataset_valid.csv')
    path_save = str(result_root / 'results_ranked_by_dataset_valid.parquet')
    shrink_ranked_result_file_size(path_load=path_load, path_save=path_save)
