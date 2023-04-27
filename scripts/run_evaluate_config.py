
from autogluon_zeroshot.simulation.ensemble_selection_config_scorer import EnsembleSelectionConfigScorer
from autogluon_zeroshot.contexts import get_context
from autogluon_zeroshot.utils import catchtime

if __name__ == '__main__':
    context_name = 'D104_F10_C608_FULL'
    benchmark_context = get_context(context_name)
    with catchtime("eval config"):
        zsc, configs_full, zeroshot_pred_proba, zeroshot_gt = benchmark_context.load(load_predictions=True)
        zsc.print_info()

        # NOTE: For speed of simulation, it is recommended backend='ray'
        #  If 'ray' isn't available, then specify 'native'.
        backend = 'ray'

        # Evaluate when using all configs
        configs = zsc.get_configs()
        if configs is not None:
            zeroshot_pred_proba.restrict_models(configs)
        datasets = zsc.get_dataset_folds()
        config_scorer = EnsembleSelectionConfigScorer.from_zsc(
            datasets=datasets,
            zeroshot_simulator_context=zsc,
            zeroshot_gt=zeroshot_gt,
            zeroshot_pred_proba=zeroshot_pred_proba,
            backend=backend,
            ensemble_size=100,  # 100 is better, but 10 allows to simulate 10x faster
        )

        score = config_scorer.score(configs)

        # With ensemble_size=100 610 datasets, 608 configs, I get
        #  Final Score: 3.8278688524590163
        print(f'Final Score: {score}')
