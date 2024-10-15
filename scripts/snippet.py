from tabrepo.repository.evaluation_repository import load_repository

# load TabRepo with 200 datasets, 3 folds and 1416 configurations
repository = load_repository("D244_F3_C1416_200")

# returns in ~2s the tensor of metrics for each dataset/fold obtained after ensembling the given configurations
metrics = repository.evaluate_ensembles(
    datasets=["abalone", "adult"],  # OpenML dataset to report results on
    folds=[0, 1, 2],  # which task to consider for each dataset
    configs=["CatBoost_r42_BAG_L1", "RandomForest_r12_BAG_L1", "NeuralNetTorch_r40_BAG_L1"],  # configs that are ensembled
    ensemble_size=40,  # maximum number of Caruana steps
)
