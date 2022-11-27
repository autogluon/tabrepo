
from autogluon.common.savers import save_pd

from autogluon_zeroshot.contexts.context_2022_10_13 import load_context_2022_10_13
from autogluon_zeroshot.simulation.sim_output import SimulationOutputGenerator


if __name__ == '__main__':
    """
    Generate an output file that is compatible with the AutoMLBenchmark results format.
    Useful to compare directly with AutoML frameworks and baselines.
    
    This script takes as input a list of configs to generate the output without cross-validation.
    """

    name = 'EnsembleAllHPO'
    zsc, configs_full, zeroshot_pred_proba, zeroshot_gt = load_context_2022_10_13(load_zeroshot_pred_proba=True)
    zsc.print_info()

    configs = zsc.get_configs()

    sog = SimulationOutputGenerator(
        zsc=zsc,
        zeroshot_gt=zeroshot_gt,
        zeroshot_pred_proba=zeroshot_pred_proba,
        backend='ray',
    )

    df_result = sog.from_portfolio(
        portfolio=configs,
        datasets=zsc.get_dataset_folds(),
        name=name,
    )

    # Use this file to compare theoretical performance to AutoGluon in separate analysis repo
    save_pd.save(path=f'zeroshot_results/zs_{name}.csv', df=df_result)
    s3_prefix = 's3://autogluon-zeroshot/config_results'
    save_pd.save(path=f'{s3_prefix}/zs_{name}.csv', df=df_result)
