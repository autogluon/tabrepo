
# key = name in AutoGluon
# val = key in AutoGluon.fit hyperparameters argument
MODEL_TYPE_DICT = {
    'LightGBM': 'GBM',
    'CatBoost': 'CAT',
    'XGBoost': 'XGB',
    'NeuralNetFastAI': 'FASTAI',
    'NeuralNetTorch': 'NN_TORCH',
    'KNeighbors': 'KNN',
    'RandomForest': 'RF',
    'ExtraTrees': 'XT',
}
