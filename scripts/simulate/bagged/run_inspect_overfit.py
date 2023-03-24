from autogluon_zeroshot.simulation.single_best_config_scorer import SingleBestConfigScorer
from autogluon_zeroshot.contexts.context_2022_12_11_bag import load_context_2022_12_11_bag
from autogluon_zeroshot.simulation.sim_runner import run_zs_simulation_debug


if __name__ == '__main__':
    zsc, configs_full, zeroshot_pred_proba, zeroshot_gt = load_context_2022_12_11_bag(
        subset='small_30'
    )
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
        config_generator_kwargs={'num_zeroshot': 20},
        backend=backend,
    )
