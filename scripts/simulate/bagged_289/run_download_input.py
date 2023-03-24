from autogluon_zeroshot.loaders import download_zs_metadata, Paths


if __name__ == '__main__':
    path_prefix = 's3://automl-benchmark-ag/aggregated/ec2/2023_03_19_zs/'
    # path_gt_name = 'zeroshot_gt_100_mb.pkl'
    # path_pred_proba_name = 'zeroshot_pred_proba_100_mb.pkl'
    #
    # download_zs_metadata(
    #     path_prefix_in=path_prefix,
    #     path_prefix_out=Paths.bagged_208_results_root,
    #     name_in_gt=path_gt_name,
    #     name_in_pred_proba=path_pred_proba_name,
    # )
