from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from autogluon.core.models import AbstractModel

if TYPE_CHECKING:
    import pandas as pd


class CustomRandomForestModel(AbstractModel):
    ag_key = "DPDT"
    ag_name = "dpdt"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator = None

    def _preprocess(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        X = super()._preprocess(X, **kwargs)
        return X.to_numpy()

    def _fit(
        self,
        X: pd.DataFrame,  # training data
        y: pd.Series,  # training labels
        # X_val=None,  # val data (unused in RF model)
        # y_val=None,  # val labels (unused in RF model)
        # time_limit=None,  # time limit in seconds (ignored in tutorial)
        num_cpus: int = 1,  # number of CPUs to use for training
        # num_gpus: int = 0,  # number of GPUs to use for training
        **kwargs,  # kwargs includes many other potential inputs, refer to AbstractModel documentation for details
    ):
        # Select model class
        if self.problem_type in ["regression"]:
            from dpdt import DPDTreeRegressor

            model_cls = DPDTreeRegressor
        else:
            from dpdt import DPDTreeClassifier

            # case for 'binary' and 'multiclass',
            model_cls = DPDTreeClassifier

        X = self.preprocess(X)
        y = self.preprocess(y)
        params = self._get_model_params()
        self.model = model_cls(**params)
        self.model.fit(X, y)

    def _set_default_params(self):
        """Default parameters for the model."""
        default_params = {
            "max_depth": 10,
            "n_jobs": -1,
            "random_state": 0,
            "cart_nodes_list": (8,3,)
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_default_auxiliary_params(self) -> dict:
        """Specifics allowed input data and that all other dtypes should be handled
        by the model-agnostic preprocessor.
        """
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = {
            "valid_raw_types": ["int", "float", "category"],
        }
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params


# def get_configs_for_custom_rf(
#     *,
#     default_config: bool = True,
#     num_random_configs: int = 1,
#     sequential_fold_fitting: bool = False,
# ):
#     """Generate the hyperparameter configurations to run for our custom random
#     forest model.

#     sequential_fold_fitting: bool = False
#         If True, the model will be configured to use sequential
#         fold fitting (better for debugging, but usually slower). This is also a good
#         idea to use on SLURM or other shared compute clusters where you want to run
#         multiple jobs on the same  node.
#         See `tabflow_slurm.run_tabarena_experiment.setup_slurm_job`  for ways to
#         optimally use sequential_fold_fitting=False on SLURM.
#     """
#     from autogluon.common.space import Int
#     from tabrepo.utils.config_utils import ConfigGenerator

#     manual_configs = [
#         {},
#     ]
#     search_space = {
#         "n_estimators": Int(4, 50),
#     }

#     gen_custom_rf = ConfigGenerator(
#         model_cls=CustomRandomForestModel,
#         manual_configs=manual_configs if default_config else None,
#         search_space=search_space,
#     )
#     experiments_lst = gen_custom_rf.generate_all_bag_experiments(
#         num_random_configs=num_random_configs
#     )

#     if sequential_fold_fitting:
#         for m_i in range(len(experiments_lst)):
#             if (
#                 "ag_args_ensemble"
#                 not in experiments_lst[m_i].method_kwargs["model_hyperparameters"]
#             ):
#                 experiments_lst[m_i].method_kwargs["model_hyperparameters"][
#                     "ag_args_ensemble"
#                 ] = {}
#             experiments_lst[m_i].method_kwargs["model_hyperparameters"][
#                 "ag_args_ensemble"
#             ]["fold_fitting_strategy"] = "sequential_local"

#     return experiments_lst