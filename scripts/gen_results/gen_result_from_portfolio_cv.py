from pathlib import Path

from autogluon.common.loaders import load_pkl
from autogluon.common.savers import save_pd

from autogluon_zeroshot.contexts import get_context
from autogluon_zeroshot.simulation.sim_output import SimulationOutputGenerator


if __name__ == '__main__':
    """
    Generate an output file that is compatible with the AutoMLBenchmark results format.
    Useful to compare directly with AutoML frameworks and baselines.
    
    This script takes as input a PortfolioCV object to produce a result that ensures no overfitting.
    """

    name = 'EnsembleCV'
    context_name = 'D104_F10_C608_FULL'
    benchmark_context = get_context(context_name)
    zsc, configs_full, zeroshot_pred_proba, zeroshot_gt = benchmark_context.load(load_predictions=True)
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
