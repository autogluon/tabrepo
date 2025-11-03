"""Run a TabArena model (tuned + ensemble) on any task."""

from __future__ import annotations

from autogluon.tabular import TabularPredictor
from data_utils import (
    get_example_data_for_task_type,
    score_for_task_type,
)

from tabarena.models.utils import get_configs_generator_from_name

task_type = "regression"
"""Task type for the model to run on.
Either "binary", "multiclass", or "regression".
"""
use_ensemble_model = True
"""If True, post-hoc ensemble the tuned configurations of the model."""
n_hyperparameter_configs = 5
"""The number of random configurations to evaluate for the model.
TabArena used 200, we use 5 for a minimal example.
"""
model_to_run = "RealMLP"
"""Select a model to run, which we automatically load in the code below.

Note: not all models are available for all task types.

The recommended options are:
    - "RealMLP"
    - "TabM"
    - "LightGBM"
    - "CatBoost"
    - "XGBoost"
    - "ModernNCA"
    - "TabPFNv2"
    - "TabICL"
    - "TorchMLP"
    - "TabDPT"
    - "EBM"
    - "FastaiMLP"
    - "ExtraTrees
    - "RandomForest"
    - "KNN"
    - "Linear"
"""
model_meta = get_configs_generator_from_name(model_name=model_to_run)
model_cls = model_meta.model_cls
hpo_configs = model_meta.generate_all_configs_lst(num_random_configs=n_hyperparameter_configs)

X_train, X_test, y_train, y_test = get_example_data_for_task_type(task_type=task_type)
train_data = X_train.copy()
train_data["target"] = y_train

# --- Tuning and ensembling a TabArena Model
# We use the AutoGluon interface for tuned (+ ensemble) models.
model_hyperparameters = {model_cls: hpo_configs}
model = TabularPredictor(
    label="target",
    eval_metric="rmse" if task_type == "regression" else "accuracy",
    problem_type=task_type,
).fit(
    train_data,
    fit_weighted_ensemble=use_ensemble_model,
    hyperparameters=model_hyperparameters,
    num_bag_folds=8,
    # Set a time limit for the tuning process. Increase this for real use.
    time_limit=360,
    # Uncomment below for testing (on SLURM).
    # ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
    #   On a bigger machine, we recommend to remove this line and potentially
    #   set `fit_strategy` to "parallel".
)
# Get validation performance
model.leaderboard(display=True)

y_pred = model.predict(data=X_test)
score_for_task_type(y_test=y_test, y_pred=y_pred, task_type=task_type)
