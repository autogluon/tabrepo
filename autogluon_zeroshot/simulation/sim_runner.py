
from .config_generator import ZeroshotConfigGeneratorCV
from .simulation_context import ZeroshotSimulatorContext


def run_zs_simulation(zsc: ZeroshotSimulatorContext, config_scorer, n_splits=10, configs=None, backend='ray') -> list:
    zs_config_generator_cv = ZeroshotConfigGeneratorCV(
        n_splits=n_splits,
        zeroshot_simulator_context=zsc,
        config_scorer=config_scorer,
        configs=configs,
        backend=backend,
    )

    results = zs_config_generator_cv.run()

    for i in range(len(results)):
        print(f'Fold {results[i]["fold"]} Selected Configs: {results[i]["selected_configs"]}')

    for i in range(len(results)):
        print(f'Fold {results[i]["fold"]} Test Score: {results[i]["score"]}')

    return results
