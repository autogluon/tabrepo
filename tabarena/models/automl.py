def generate_autogluon_experiments():
    from tabarena.benchmark.experiment.experiment_constructor import AGExperiment
    methods = [
        AGExperiment(
            name="AutoGluon_bq_4h8c",
            fit_kwargs=dict(
                time_limit=14400,
                presets="best",
            ),
        ),
        AGExperiment(
            name="AutoGluon_bq_1h8c",
            fit_kwargs=dict(
                time_limit=3600,
                presets="best",
            ),
        ),
        AGExperiment(
            name="AutoGluon_bq_5m8c",
            fit_kwargs=dict(
                time_limit=300,
                presets="best",
            ),
        ),
    ]
    return methods
