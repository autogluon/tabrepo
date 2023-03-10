from autogluon.common.loaders import load_pd
from autogluon.common.savers import save_pd


# FIXME: Make this less hard-coded
if __name__ == '__main__':
    local_path_prefix = './data/results/bagged_208/'
    s3_path_prefix = 's3://automl-benchmark-ag/aggregated/ec2/2023_02_27_zs/'

    file_list = [
        'results_ranked_by_dataset_valid.csv',
        'results_ranked_valid.csv',
        'openml_ag_2023_02_27_zs_models.csv',
    ]

    for f in file_list:
        d = load_pd.load(path=f'{local_path_prefix}{f}')
        save_pd.save(path=f'{s3_path_prefix}{f}', df=d)
