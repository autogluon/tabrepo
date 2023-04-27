# File to check that we get the same error as run_evaluate_config.py. It can be removed.
from autogluon_zeroshot.contexts import get_context
from autogluon_zeroshot.utils import catchtime
from scripts.method_comparison.evaluate_ensemble import evaluate_ensemble

if __name__ == '__main__':
    bag = True
    if bag:
        context_name = 'BAG_D104_F10_C608_FULL'
    else:
        context_name = 'D104_F10_C608_FULL'
    benchmark_context = get_context(context_name)

    with catchtime("eval"):
        with catchtime("load"):
            zsc, configs_full, zeroshot_pred_proba, zeroshot_gt = benchmark_context.load(load_predictions=False)
        configs = zsc.get_configs()
        datasets = zsc.get_dataset_folds()

        train_error, _ = evaluate_ensemble(
            configs=configs,
            train_datasets=datasets,
            test_datasets=[],
            ensemble_size=1,
            bag=bag,
            backend="native",
        )

        # With ensemble_size=100 610 datasets, 608 configs, I get
        #  Final Score: 3.8311475409836064
        print(f'Final Score: {train_error, _}')
