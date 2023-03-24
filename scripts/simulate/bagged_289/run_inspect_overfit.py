from autogluon_zeroshot.simulation.single_best_config_scorer import SingleBestConfigScorer
from autogluon_zeroshot.contexts.context_2023_03_19_bag_289 import load_context_2023_03_19_bag_289
from autogluon_zeroshot.simulation.sim_runner import run_zs_simulation_debug


if __name__ == '__main__':
    # import boto3
    #
    # s3 = boto3.resource('s3')
    #
    # path_prefix = 'plots/20230313_021045/'
    #
    # s3_bucket = 'autogluon-zeroshot'
    # s3_prefix = f'sim_output/bagged_208/{path_prefix}'
    #
    # for f in ['overfit_delta_comparison.png', 'train_test_comparison.png']:
    #     s3.Bucket(s3_bucket).upload_file(f"{path_prefix}{f}", f"{s3_prefix}{f}")

    zsc, configs_full, zeroshot_pred_proba, zeroshot_gt = load_context_2023_03_19_bag_289()
    zsc.print_info()

    # NOTE: For speed of simulation, it is recommended backend='ray'
    backend = 'ray'

    # configs = get_configs_small()
    configs = None

    datasets = zsc.get_dataset_folds()

    config_scorer = SingleBestConfigScorer.from_zsc(
        zeroshot_simulator_context=zsc,
        datasets=datasets,
    )

    run_zs_simulation_debug(
        zsc=zsc,
        config_scorer=config_scorer,
        n_splits=5,
        n_repeats=10,
        configs=configs,
        config_generator_kwargs={
            'num_zeroshot': 20,
            # 'removal_stage': True,
        },
        backend=backend,
    )
