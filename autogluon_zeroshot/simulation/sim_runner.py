
from .config_generator import ZeroshotConfigGeneratorCV
from .simulation_context import ZeroshotSimulatorContext
from ..portfolio import PortfolioCV


def run_zs_simulation(zsc: ZeroshotSimulatorContext, config_scorer, n_splits=10, config_generator_kwargs=None, configs=None, backend='ray') -> PortfolioCV:
    zs_config_generator_cv = ZeroshotConfigGeneratorCV(
        n_splits=n_splits,
        zeroshot_simulator_context=zsc,
        config_scorer=config_scorer,
        config_generator_kwargs=config_generator_kwargs,
        configs=configs,
        backend=backend,
    )

    portfolio_cv = zs_config_generator_cv.run()

    portfolios = portfolio_cv.portfolios
    for i in range(len(portfolios)):
        print(f'Fold {portfolios[i].fold} Selected Configs: {portfolios[i].configs}')

    for i in range(len(portfolios)):
        print(f'Fold {portfolios[i].fold} Test Score: {portfolios[i].test_score}')

    return portfolio_cv
