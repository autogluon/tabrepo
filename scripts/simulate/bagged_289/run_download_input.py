from autogluon_zeroshot.loaders import download_zs_metadata, Paths


if __name__ == '__main__':
    path_prefix = 's3://automl-benchmark-ag/aggregated/ec2/2023_03_19_zs/'
    # TODO: Include download_s3_folder when it is merged into AG
    # from autogluon.common.utils.s3_utils import download_s3_folder
    # from autogluon.common.utils.log_utils import set_logger_verbosity
    # set_logger_verbosity(2)
    #
    # download_s3_folder(bucket='automl-benchmark-ag',
    #                    prefix='aggregated/ec2/2023_03_19_zs/zs_input/bagged_289/',
    #                    local_path=str(Paths.bagged_289_results_root),
    #                    error_if_exists=False,
    #                    delete_if_exists=False,
    #                    dry_run=False)

    for zs_mb_size in [
        10,
        50,
    ]:
        path_gt_name = f'zeroshot_gt_{zs_mb_size}_mb.pkl'
        path_pred_proba_name = f'zeroshot_pred_proba_{zs_mb_size}_mb.pkl'

        download_zs_metadata(
            path_prefix_in=path_prefix,
            path_prefix_out=Paths.bagged_289_results_root,
            name_in_gt=path_gt_name,
            name_in_pred_proba=path_pred_proba_name,
        )
