from autogluon_zeroshot.loaders import download_zs_metadata, Paths


if __name__ == '__main__':
    path_prefix = 's3://automl-benchmark-ag/aggregated/ec2/2022_10_13_zs/'
    path_gt_name = 'zeroshot_gt_2022_10_13_zs.pkl'
    path_pred_proba_name = 'zeroshot_pred_proba_2022_10_13_zs.pkl'

    # WARNING: This file is 17 GB on disk. Ensure your machine has >100 GB of memory to avoid crashing.
    #  Additionally, ensure your internet connection is strong, otherwise this will take an extremely long time.
    download_zs_metadata(
        path_prefix_in=path_prefix,
        path_prefix_out=Paths.all_v3_results_root,
        name_in_gt=path_gt_name,
        name_in_pred_proba=path_pred_proba_name,
    )
