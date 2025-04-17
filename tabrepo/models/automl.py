def generate_autogluon_experiments():
    from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
    hyperparameters_default = get_hyperparameter_config("zeroshot")

    from tabrepo.benchmark.models.ag import RealMLPModel, TabICLModel, ExplainableBoostingMachineModel
    hyperparameters_new = {
        TabICLModel: [
            {"n_estimators": 1, "ag_args": {"name_suffix": "_c1", "priority": 999}},
        ],
        ExplainableBoostingMachineModel: [
            {"ag_args": {"name_suffix": "_c1", "priority": 998}},
        ],
        RealMLPModel: [
            {"ag_args": {"name_suffix": "_c1", "priority": 997}},
        ],
    }
    import copy
    hyperparameters_custom = copy.deepcopy(hyperparameters_default)
    hyperparameters_custom.update(hyperparameters_new)
    from tabrepo.benchmark.experiment.experiment_constructor import AGExperiment
    # method = AGExperiment(
    #     name="AutoGluon_bq_1h8c_extras",
    #     fit_kwargs=dict(
    #         time_limit=3600,
    #         presets="best",
    #         hyperparameters=hyperparameters_new,
    #     ),
    # )
    method = AGExperiment(
        name="AutoGluon_bq_5m8c_extras",
        fit_kwargs=dict(
            time_limit=300,
            presets="best",
            hyperparameters=hyperparameters_custom,
        ),
    )
    methods = [
        method,
        AGExperiment(
            name="AutoGluon_bq_5m8c",
            fit_kwargs=dict(
                time_limit=300,
                presets="best",
            ),
        ),
    ]
    return methods
