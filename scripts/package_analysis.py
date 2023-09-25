import pandas as pd
from baseline_comparison.plot_utils import save_latex_table


# TODO: Could potentially add a Reference column for paper citation (as done in AMLB paper)
metadata = {
    "AutoGluon": {
        "benchmarked": "0.8.2",
        "latest": "0.8.2",
        "package": "autogluon",
    },
    "auto-sklearn": {
        "benchmarked": "0.15.0",
        "latest": "0.15.0",
        "package": "auto-sklearn",
    },
    "auto-sklearn 2": {
        "benchmarked": "0.15.0",
        "latest": "0.15.0",
        "package": "auto-sklearn",
    },
    "FLAML": {
        "benchmarked": "1.2.4",
        "latest": "2.1.0",
        "package": "flaml",
    },
    "H2O AutoML": {
        "benchmarked": "3.40.0.4",
        "latest": "3.42.0.3",
        "package": "h2o",
    },
    "LightAutoML": {
        "benchmarked": "0.3.7.3",
        "latest": "0.3.7.3",
        "package": "lightautoml",
    },
}

df = pd.DataFrame(metadata).T
df.index.name = 'framework'
df = df.reset_index(drop=False)
latex_kwargs = dict(
    index=False,
)

print(df)

save_latex_table(df=df, title="automl_versions", latex_kwargs=latex_kwargs)


metadata = {
    "LightGBM": {
        "benchmarked": "3.3.5",
        "latest": "4.0.0",
        "package": "lightgbm",
    },
    "XGBoost": {
        "benchmarked": "1.7.6",
        "latest": "2.0.0",
        "package": "xgboost",
    },
    "CatBoost": {
        "benchmarked": "1.2.1",
        "latest": "1.2.2",
        "package": "catboost",
    },
    "RandomForest": {
        "benchmarked": "1.1.1",
        "latest": "1.3.1",
        "package": "scikit-learn",
    },
    "ExtraTrees": {
        "benchmarked": "1.1.1",
        "latest": "1.3.1",
        "package": "scikit-learn",
    },
    "MLP": {
        "benchmarked": "2.0.1",
        "latest": "2.0.1",
        "package": "torch",
    },
}


df = pd.DataFrame(metadata).T
df.index.name = 'model'
df = df.reset_index(drop=False)
latex_kwargs = dict(
    index=False,
)

print(df)

save_latex_table(df=df, title="model_versions", latex_kwargs=latex_kwargs)
