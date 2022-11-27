from pathlib import Path

from autogluon.common.loaders import load_pkl
from autogluon.common.savers import save_pd

from autogluon_zeroshot.contexts.context_2022_10_13 import load_context_2022_10_13
from autogluon_zeroshot.simulation.sim_output import SimulationOutputGenerator


if __name__ == '__main__':
    """
    Generate an output file that is compatible with the AutoMLBenchmark results format.
    Useful to compare directly with AutoML frameworks and baselines.
    
    This script takes as input a PortfolioCV object to produce a result that ensures no overfitting.
    """

    name = 'EnsembleCV'
    zsc, configs_full, zeroshot_pred_proba, zeroshot_gt = load_context_2022_10_13(load_zeroshot_pred_proba=True)
    zsc.print_info()

    sog = SimulationOutputGenerator(
        zsc=zsc,
        zeroshot_gt=zeroshot_gt,
        zeroshot_pred_proba=zeroshot_pred_proba,
        backend='ray',
    )

    results = load_pkl.load(path=str(Path(__file__).parent.parent / 'sim_results' / 'ensemble_result.pkl'))
    df_result = sog.from_portfolio_cv(portfolio_cv=results, name=name)

    # Use this file to compare theoretical performance to AutoGluon in separate analysis repo
    save_pd.save(path=f'zeroshot_results/zs_{name}.csv', df=df_result)
    s3_prefix = 's3://autogluon-zeroshot/config_results'
    save_pd.save(path=f'{s3_prefix}/zs_{name}.csv', df=df_result)
