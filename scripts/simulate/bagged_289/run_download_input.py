from autogluon.common.utils.log_utils import set_logger_verbosity
from autogluon.common.utils.s3_utils import download_s3_folder  # Note: this requires latest mainline AutoGluon

from autogluon_zeroshot.loaders import download_zs_metadata, Paths


if __name__ == '__main__':
    dry_run = True  # Set to False to download files

    if dry_run:
        print(f'NOTE: Files will not be downloaded as `dry_run=True`.\n'
              f'This will log what files will be downloaded instead.\n'
              f'Set `dry_run=False` to download the files.')

    path_prefix = 's3://automl-benchmark-ag/aggregated/ec2/2023_03_19_zs/'
    set_logger_verbosity(2)
    download_s3_folder(bucket='automl-benchmark-ag',
                       prefix='aggregated/ec2/2023_03_19_zs/zs_input/bagged_289/',
                       local_path=str(Paths.bagged_289_results_root),
                       error_if_exists=False,
                       delete_if_exists=False,
                       dry_run=dry_run)

    download_s3_folder(bucket='automl-benchmark-ag',
                       prefix='aggregated/ec2/2023_03_19_zs/zs_input/automl_289/',
                       local_path=str(Paths.automl_289_results_root),
                       error_if_exists=False,
                       delete_if_exists=False,
                       dry_run=dry_run)

    for zs_mb_size in [
        10,
        50,
    ]:
        download_s3_folder(bucket='automl-benchmark-ag',
                           prefix='aggregated/ec2/2023_03_19_zs/',
                           local_path=str(Paths.bagged_289_results_root),
                           suffix=f'{zs_mb_size}_mb.pkl',
                           error_if_exists=False,
                           delete_if_exists=False,
                           dry_run=dry_run)
