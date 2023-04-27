from .context import BenchmarkContext
from ..loaders import Paths


_s3_download_map = {
    "results/bagged_289/608/results_ranked_by_dataset_valid.csv": "s3://automl-benchmark-ag/aggregated/ec2/2023_03_19_zs/zs_input/bagged_289/608/results_ranked_by_dataset_valid.csv",
    "results/bagged_289/608/results_ranked_by_dataset_valid.parquet": "s3://automl-benchmark-ag/aggregated/ec2/2023_03_19_zs/zs_input/bagged_289/608/results_ranked_by_dataset_valid.parquet",
    "results/bagged_289/608/results_ranked_valid.csv": "s3://automl-benchmark-ag/aggregated/ec2/2023_03_19_zs/zs_input/bagged_289/608/results_ranked_valid.csv",
    "results/bagged_289/608/results_ranked_valid.parquet": "s3://automl-benchmark-ag/aggregated/ec2/2023_03_19_zs/zs_input/bagged_289/608/results_ranked_valid.parquet",
    "results/bagged_289/openml_ag_2023_03_19_zs_models.csv": "s3://automl-benchmark-ag/aggregated/ec2/2023_03_19_zs/zs_input/bagged_289/openml_ag_2023_03_19_zs_models.csv",
    "results/bagged_289/openml_ag_2023_03_19_zs_models.parquet": "s3://automl-benchmark-ag/aggregated/ec2/2023_03_19_zs/zs_input/bagged_289/openml_ag_2023_03_19_zs_models.parquet",
    "results/bagged_289/results_ranked_by_dataset_valid.csv": "s3://automl-benchmark-ag/aggregated/ec2/2023_03_19_zs/zs_input/bagged_289/results_ranked_by_dataset_valid.csv",
    "results/bagged_289/results_ranked_valid.csv": "s3://automl-benchmark-ag/aggregated/ec2/2023_03_19_zs/zs_input/bagged_289/results_ranked_valid.csv",
    "results/automl_289/results_ranked_by_dataset_valid.csv": "s3://automl-benchmark-ag/aggregated/ec2/2023_03_19_zs/zs_input/automl_289/results_ranked_by_dataset_valid.csv",
    "results/automl_289/results_ranked_valid.csv": "s3://automl-benchmark-ag/aggregated/ec2/2023_03_19_zs/zs_input/automl_289/results_ranked_valid.csv",
    "results/bagged_289/zeroshot_gt_10_mb.pkl": "s3://automl-benchmark-ag/aggregated/ec2/2023_03_19_zs/zeroshot_gt_10_mb.pkl",
    "results/bagged_289/zeroshot_pred_proba_10_mb.pkl": "s3://automl-benchmark-ag/aggregated/ec2/2023_03_19_zs/zeroshot_pred_proba_10_mb.pkl",
    "results/bagged_289/zeroshot_gt_50_mb.pkl": "s3://automl-benchmark-ag/aggregated/ec2/2023_03_19_zs/zeroshot_gt_50_mb.pkl",
    "results/bagged_289/zeroshot_pred_proba_50_mb.pkl": "s3://automl-benchmark-ag/aggregated/ec2/2023_03_19_zs/zeroshot_pred_proba_50_mb.pkl",
}
_s3_download_map = {Paths.rel_to_abs(k, relative_to=Paths.data_root): v for k, v in _s3_download_map.items()}


_path_bagged_root = Paths.bagged_289_results_root

_bag_289_result_paths = dict(
    result=str(_path_bagged_root / "608/results_ranked_valid.parquet"),
    results_by_dataset=str(_path_bagged_root / "608/results_ranked_by_dataset_valid.parquet"),
    raw=str(_path_bagged_root / "openml_ag_2023_03_19_zs_models.parquet"),
    comparison=str(Paths.automl_289_results_root / "results_ranked_by_dataset_valid.csv"),
)

_task_metadata_289_path = dict(
    task_metadata=str(Paths.data_root / "metadata" / "task_metadata_289.csv"),
)

_task_metadata_244_path = dict(
    task_metadata=str(Paths.data_root / "metadata" / "task_metadata_244.csv"),
)

_bag_289_zs_50_mb_path = dict(
    zs_pp=str(_path_bagged_root / f'zeroshot_pred_proba_50_mb.pkl'),
    zs_gt=str(_path_bagged_root / f'zeroshot_gt_50_mb.pkl'),
)

_bag_289_zs_10_mb_path = dict(
    zs_pp=str(_path_bagged_root / f'zeroshot_pred_proba_10_mb.pkl'),
    zs_gt=str(_path_bagged_root / f'zeroshot_gt_10_mb.pkl'),
)

context_bag_244_bench_50_mb: BenchmarkContext = BenchmarkContext.from_paths(
    name='BAG_D244_F10_C608_FULL',
    description='Bagged results from 244 datasets (non-trivial), 10-fold CV, 608 configs. '
                '(130 dense result datasets w/ ZPP, 137 w/o)',
    date='2023_03_19',
    folds=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    s3_download_map=_s3_download_map,
    **_bag_289_result_paths,
    **_bag_289_zs_50_mb_path,
    **_task_metadata_244_path,
)

context_bag_244_bench_10_mb: BenchmarkContext = BenchmarkContext.from_paths(
    name='BAG_D244_F10_C608_MEDIUM',
    description='Bagged results from 244 datasets (non-trivial), 10-fold CV, 608 configs. '
                '(105 dense result datasets w/ ZPP, 137 w/o)',
    date='2023_03_19',
    folds=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    s3_download_map=_s3_download_map,
    **_bag_289_result_paths,
    **_bag_289_zs_10_mb_path,
    **_task_metadata_244_path,
)

context_bag_289_bench_50_mb: BenchmarkContext = BenchmarkContext.from_paths(
    name='BAG_D279_F10_C608_FULL',
    description='(NOTE: Use BAG_D244 context instead for improved results!! This context is outdated!) '
                'Bagged results from 279 datasets (includes trivial datasets), 10-fold CV, 608 configs. ',
    date='2023_03_19',
    folds=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    s3_download_map=_s3_download_map,
    **_bag_289_result_paths,
    **_bag_289_zs_50_mb_path,
    **_task_metadata_289_path,
)

context_bag_289_bench_10_mb: BenchmarkContext = BenchmarkContext.from_paths(
    name='BAG_D279_F10_C608_MEDIUM',
    description='(NOTE: Use BAG_D244 context instead for improved results!! This context is outdated!) '
                'Bagged results from 279 datasets (includes trivial datasets), 10-fold CV, 608 configs. ',
    date='2023_03_19',
    folds=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    s3_download_map=_s3_download_map,
    **_bag_289_result_paths,
    **_bag_289_zs_10_mb_path,
    **_task_metadata_289_path,
)


def get_configs_default():
    autogluon_configs = [
        'CatBoost_c1',
        'LightGBM_c1',
        'LightGBM_c2',
        'LightGBM_c3',
        'NeuralNetFastAI_c1',
        'RandomForest_c1',
        'ExtraTrees_c1',
    ]
    autogluon_configs = [a + '_BAG_L1' for a in autogluon_configs]
    return autogluon_configs


def get_configs_small(num_per_type=12):
    small_extra_configs = []
    for m in [
        'LightGBM',
        'CatBoost',
        'RandomForest',
        'ExtraTrees',
        'NeuralNetFastAI',
    ]:
        for i in range(1, num_per_type):
            small_extra_configs.append(m + f'_r{i}_BAG_L1')
    small_configs = get_configs_default() + small_extra_configs
    return small_configs
