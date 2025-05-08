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
        num_cpus: int = 1,
        num_gpus: int = 0,
        **kwargs,
    ):
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        from tabpfn.model.loading import resolve_model_path
        from torch.cuda import is_available

        ag_params = self._get_ag_params()
        max_classes = ag_params.get("max_classes")
        is_classification = self.problem_type in ["binary", "multiclass"]

        if is_classification:
            if max_classes is not None and self.num_classes > max_classes:
                raise AssertionError(
                    f"Max allowed classes for the model is {max_classes}, but found {self.num_classes} classes.",
                )

            model_base = TabPFNClassifier
        else:
            model_base = TabPFNRegressor

        device = "cuda" if num_gpus != 0 else "cpu"
        if (device == "cuda") and (not is_available()):
            # FIXME: warn instead and switch to CPU.
            raise AssertionError(
                "Fit specified to use GPU, but CUDA is not available on this machine. "
                "Please switch to CPU usage instead.",
            )

        X = self.preprocess(X, is_train=True)

        hps = self._get_model_params()
        hps["device"] = device
        hps["n_jobs"] = num_cpus
        hps["random_state"] = 42  # TODO: get seed from AutoGluon.
        hps["categorical_features_indices"] = self._cat_indices
        hps["ignore_pretraining_limits"] = True  # to ignore warnings and size limits

        _, model_dir, _, _ = resolve_model_path(
            model_path=None,
            which="classifier" if is_classification else "regressor",
        )
        if is_classification:
            if "classification_model_path" in hps:
                hps["model_path"] = model_dir / hps.pop("classification_model_path")
            if "regression_model_path" in hps:
                del hps["regression_model_path"]
        else:
            if "regression_model_path" in hps:
                hps["model_path"] = model_dir / hps.pop("regression_model_path")
            if "classification_model_path" in hps:
                del hps["classification_model_path"]

        # Resolve inference_config
        inference_config = {
            _k: v for k, v in hps.items() if k.startswith("inference_config/") and (_k := k.split("/")[-1])
        }
        if inference_config:
            hps["inference_config"] = inference_config
        for k in list(hps.keys()):
            if k.startswith("inference_config/"):
                del hps[k]

        # Resolve model_type
        n_ensemble_repeats = hps.pop("n_ensemble_repeats", None)
        model_is_rf_pfn = hps.pop("model_type", "no") == "dt_pfn"
        if model_is_rf_pfn:
            from tabrepo.benchmark.models.ag.tabpfnv2.rfpfn import (
                RandomForestTabPFNClassifier,
                RandomForestTabPFNRegressor,
            )

            hps["n_estimators"] = 1
            rf_model_base = RandomForestTabPFNClassifier if is_classification else RandomForestTabPFNRegressor
            self.model = rf_model_base(
                tabpfn=model_base(**hps),
                categorical_features=self._cat_indices,
                n_estimators=n_ensemble_repeats,
            )
        else:
            if n_ensemble_repeats is not None:
                hps["n_estimators"] = n_ensemble_repeats
            self.model = model_base(**hps)

        self.model = self.model.fit(
            X=X,
            y=y,
        )

    def _get_default_resources(self) -> tuple[int, int]:
        from autogluon.common.utils.resource_utils import ResourceManager
        from torch.cuda import is_available

        num_cpus = ResourceManager.get_cpu_count_psutil()
        num_gpus = 1 if is_available() else 0
        return num_cpus, num_gpus

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
