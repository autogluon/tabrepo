from __future__ import annotations

from typing import TYPE_CHECKING

from autogluon.core.models import AbstractModel
from autogluon.features.generators import LabelEncoderFeatureGenerator

if TYPE_CHECKING:
    import pandas as pd


# FIXME: why so slow?
class TabPFNV2Model(AbstractModel):
    ag_key = "TABPFNV2"
    ag_name = "TabPFNv2"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator = None
        self._cat_features = None

    def _preprocess(self, X: pd.DataFrame, is_train=False, **kwargs) -> pd.DataFrame:
        X = super()._preprocess(X, **kwargs)
        self._cat_indices = []

        if is_train:
            # X will be the training data.
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)

        # This converts categorical features to numeric via stateful label encoding.
        if self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)

            # Detect/set cat features and indices
            if self._cat_features is None:
                self._cat_features = self._feature_generator.features_in[:]
            self._cat_indices = [X.columns.get_loc(col) for col in self._cat_features]

        return X

    # FIXME: What is the minimal model artifact?
    #  If zeroshot, maybe we don't save weights for each fold in bag and instead load from a single weights file?
    # FIXME: Crashes during model download if bagging with parallel fit.
    #  Consider adopting same download logic as TabPFNMix which doesn't crash during model download.
    # FIXME: Maybe support child_oof somehow with using only one model and being smart about inference time?
    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        num_cpus: int = 1,
        num_gpus: int = 0,
        **kwargs,
    ):
        import torch
        from tabpfn import TabPFNClassifier, TabPFNRegressor

        ag_params = self._get_ag_params()
        max_classes = ag_params.get("max_classes")

        if self.problem_type in ["binary", "multiclass"]:
            if max_classes is not None and self.num_classes > max_classes:
                raise AssertionError(
                    f"Max allowed classes for the model is {max_classes}, but found {self.num_classes} classes.",
                )

            model_cls = TabPFNClassifier
        else:
            model_cls = TabPFNRegressor

        device = "cuda" if num_gpus != 0 else "cpu"
        if (device == "cuda") and (not torch.cuda.is_available()):
            # FIXME: warn instead and switch to CPU.
            raise AssertionError(
                "Fit specified to use GPU, but CUDA is not available on this machine. "
                "Please switch to CPU usage instead.",
            )

        X = self.preprocess(X, is_train=True)

        hps = self._get_model_params()
        self.model = model_cls(
            device=device,
            n_jobs=num_cpus,
            random_state=42,  # TODO: get seed from AutoGluon.
            categorical_features_indices=self._cat_indices,
            ignore_pretraining_limits=True,  # to ignore warnings and size limits
            **hps,
        )

        self.model = self.model.fit(
            X=X,
            y=y,
        )

    # FIXME: We have to assume the models are downloaded beforehand / as part of the installation.
    # @classmethod
    # def _get_default_ag_args_ensemble(cls, **kwargs) -> dict:
    #     default_ag_args_ensemble = super()._get_default_ag_args_ensemble(**kwargs)
    #     # FIXME: Will raise an exception if the model isn't downloaded
    #     extra_ag_args_ensemble = {
    #        "fold_fitting_strategy": "sequential_local",  # FIXME: Comment out after debugging for large speedup
    #     }
    #     default_ag_args_ensemble.update(extra_ag_args_ensemble)
    #     return default_ag_args_ensemble

    # FIXME: unsure what the purpose of this code is
    # def _get_default_resources(self) -> tuple[int, int]:
    #     # logical=False is faster in training
    #     num_cpus = ResourceManager.get_cpu_count_psutil(logical=False)
    #     num_gpus = 0
    #     return num_cpus, num_gpus

    def _set_default_params(self):
        default_params = {}
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["binary", "multiclass", "regression"]

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        default_auxiliary_params.update(
            {
                "max_classes": 10,
            },
        )
        return default_auxiliary_params

    def _ag_params(self) -> set:
        return {"max_classes"}

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}
