import pandas as pd

from tabarena import get_context, list_contexts, EvaluationRepository


if __name__ == '__main__':
    context_names = list_contexts()
    print(f"Available Contexts: {context_names}")

    for context_name in context_names:
        context = get_context(context_name)
        print(f"\t{context_name}\t: {context.description}")

    context_name = "D244_F3_C1530_30"
    context = get_context(name=context_name)

    repo: EvaluationRepository = EvaluationRepository.from_context(version=context_name, cache=True)

    repo.print_info()

    datasets = repo.datasets()
    print(f"Datasets: {datasets}")

    dataset = "Australian"
    dataset_info = repo.dataset_info(dataset=dataset)
    print(f"Dataset Info    : {dataset_info}")

    dataset_metadata = repo.dataset_metadata(dataset=dataset)
    print(f"Dataset Metadata: {dataset_metadata}")

    configs = repo.configs()
    print(f"Configs (first 10): {configs[:10]}")

    config = "CatBoost_r1_BAG_L1"
    config_type = repo.config_type(config=config)
    config_hyperparameters = repo.config_hyperparameters(config=config)

    # You can pass the below autogluon_hyperparameters into AutoGluon's TabularPredictor.fit call to fit the specific config on a new dataset:
    # from autogluon.tabular import TabularPredictor
    # predictor = TabularPredictor(...).fit(..., hyperparameters=autogluon_hyperparameters)
    autogluon_hyperparameters = repo.autogluon_hyperparameters_dict(configs=[config])
    print(f"Config Info:\n"
          f"\t                     Name: {config}\n"
          f"\t                     Type: {config_type}\n"
          f"\t          Hyperparameters: {config_hyperparameters}\n"
          f"\tAutoGluon Hyperparameters: {autogluon_hyperparameters}\n")

    metrics = repo.metrics(datasets=["Australian", "balance-scale"], configs=["CatBoost_r1_BAG_L1", "LightGBM_r41_BAG_L1"])
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(f"Config Metrics Example:\n{metrics}")

    predictions_test = repo.predict_test(dataset=dataset, fold=0, config=config)
    print(f"Predictions Test (config={config}, dataset={dataset}, fold=0):\n{predictions_test}")

    y_test = repo.labels_test(dataset=dataset, fold=0)
    print(f"Ground Truth Test (dataset={dataset}, fold=0):\n{y_test}")

    predictions_val = repo.predict_val(dataset=dataset, fold=0, config=config)
    print(f"Predictions Val (config={config}, dataset={dataset}, fold=0):\n{predictions_val[:10]}")

    y_val = repo.labels_val(dataset=dataset, fold=0)
    print(f"Ground Truth Val (dataset={dataset}, fold=0):\n{y_val[:10]}")

    df_result, df_ensemble_weights = repo.evaluate_ensemble(dataset=dataset, fold=0, configs=configs, ensemble_size=100)
    print(f"Ensemble result:\n{df_result}")

    df_ensemble_weights_mean_sorted = df_ensemble_weights.mean(axis=0).sort_values(ascending=False)
    print(f"Top 10 highest mean ensemble weight configs:\n{df_ensemble_weights_mean_sorted.head(10)}")

    # Advanced
    """
    from tabarena.repository import EvaluationRepositoryZeroshot
    repo: EvaluationRepositoryZeroshot = repo.to_zeroshot()
    results_cv = repo.simulate_zeroshot(num_zeroshot=5, n_splits=2)
    # FIXME: Incorrect infer_time
    df_results = repo.generate_output_from_portfolio_cv(portfolio_cv=results_cv, name="quickstart")
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(df_results)

    df_baselines = repo._zeroshot_context.df_baselines
    df_concat = pd.concat([df_results, df_baselines.drop(columns=["task"])], ignore_index=True)

    from autogluon_benchmark.evaluation.evaluate_results import evaluate
    results_ranked_valid, results_ranked_by_dataset_valid, results_ranked_all, results_ranked_by_dataset_all, results_pairs_merged_dict = evaluate(
        results_raw=df_concat, frameworks_compare_vs_all=["quickstart"],
        columns_to_agg_extra=['time_infer_s']
    )
    """
