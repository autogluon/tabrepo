from __future__ import annotations

import logging
import os
import sys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.models import AbstractModel
from autogluon.features.generators import LabelEncoderFeatureGenerator

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class LimiXModel(AbstractModel):
    """Rel: https://github.com/limix-ldm/LimiX."""

    ag_key = "LIMIX"
    ag_name = "LimiX"
    _DEFAULT_CHECKPOINT_PATH = "LimiX-16M.ckpt"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator = None
        self._cat_features = None
        self._cat_indices = None

    def _preprocess(self, X: pd.DataFrame, is_train=False, **kwargs) -> pd.DataFrame:
        X = super()._preprocess(X, **kwargs)

        if is_train:
            self._cat_indices = []

            # X will be the training data°.
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)

        # This converts categorical features to numeric via stateful label encoding.
        if self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(
                X=X
            )

            if is_train:
                # Detect/set cat features and indices
                if self._cat_features is None:
                    self._cat_features = self._feature_generator.features_in[:]
                self._cat_indices = [
                    X.columns.get_loc(col) for col in self._cat_features
                ]

        return X

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_cpus: int = 1,
        num_gpus: int = 0,
        verbosity: int = 2,
        **kwargs,
    ):
        import torch

        from tabrepo.benchmark.models.ag.limix.LimiX.inference.predictor import (
            LimiXPredictor,
        )

        is_classification = self.problem_type in ["binary", "multiclass"]
        device = "cuda" if num_gpus != 0 else "cpu"
        if (device == "cuda") and (not torch.cuda.is_available()):
            # FIXME: warn instead and switch to CPU.
            raise AssertionError(
                "Fit specified to use GPU, but CUDA is not available on this machine. "
                "Please switch to CPU usage instead.",
            )

        X = self.preprocess(X, is_train=True)

        cls_config_default_config = (
            Path(__file__).parent / "LimiX" / "config" / "cls_default_noretrieval.json"
        )
        reg_config_default_config = (
            Path(__file__).parent / "LimiX" / "config" / "reg_default_noretrieval.json"
        )
        inference_config = (
            cls_config_default_config
            if is_classification
            else reg_config_default_config
        )

        hps = self._get_model_params()
        hps["device"] = device

        self.model = LimiXPredictor(
            X_train=X,
            y_train=y,
            seed=self.random_seed,
            model_path=self.download_model(),
            categorical_features_indices=self._cat_indices,
            inference_config=str(inference_config.resolve()),
            task_type="Classification" if is_classification else "Regression",
            **hps,
        )

    def _get_default_resources(self) -> tuple[int, int]:
        # Use only physical cores for better performance based on benchmarks
        num_cpus = ResourceManager.get_cpu_count(only_physical_cores=True)

        num_gpus = min(1, ResourceManager.get_gpu_count_torch(cuda_only=True))

        return num_cpus, num_gpus

    def get_minimum_resources(
        self, is_gpu_available: bool = False
    ) -> dict[str, int | float]:
        return {
            "num_cpus": 1,
            "num_gpus": 1 if is_gpu_available else 0,
        }

    @staticmethod
    def download_model():
        from huggingface_hub import hf_hub_download

        model_dir = _user_cache_dir(platform=sys.platform, appname="limix")
        model_dir.mkdir(exist_ok=True, parents=True)

        final_model_path = model_dir / LimiXModel._DEFAULT_CHECKPOINT_PATH

        if not final_model_path.exists():
            model_file = hf_hub_download(
                repo_id="stableai-org/LimiX-16M",
                filename=LimiXModel._DEFAULT_CHECKPOINT_PATH,
                local_dir=str(model_dir),
            )
            assert str(final_model_path) == model_file
        return str(final_model_path)

    def _set_default_params(self):
        default_params = {}
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_random_seed_from_hyperparameters(
        self, hyperparameters: dict
    ) -> int | None | str:
        return hyperparameters.get("seed", "N/A")

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["binary", "multiclass", "regression"]

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        default_auxiliary_params.update(
            {
                "max_classes": 10,
            }
        )
        return default_auxiliary_params

    @classmethod
    def _get_default_ag_args_ensemble(cls, **kwargs) -> dict:
        """Set fold_fitting_strategy to sequential_local,
        as parallel folding crashes if model weights aren't pre-downloaded.
        """
        default_ag_args_ensemble = super()._get_default_ag_args_ensemble(**kwargs)
        extra_ag_args_ensemble = {
            "fold_fitting_strategy": "sequential_local",
            "refit_folds": True,
        }
        default_ag_args_ensemble.update(extra_ag_args_ensemble)
        return default_ag_args_ensemble

    @classmethod
    def _class_tags(cls):
        return {"can_estimate_memory_usage_static": False}

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}


def _user_cache_dir(platform: str, appname: str = "tabpfn") -> Path:
    use_instead_path = (Path.cwd() / ".tabpfn_models").resolve()

    # https://docs.python.org/3/library/sys.html#sys.platform
    if platform == "win32":
        # Honestly, I don't want to do what `platformdirs` does:
        # https://github.com/tox-dev/platformdirs/blob/b769439b2a3b70769a93905944a71b3e63ef4823/src/platformdirs/windows.py#L252-L265
        APPDATA_PATH = os.environ.get("APPDATA", "")
        if APPDATA_PATH.strip() != "":
            return Path(APPDATA_PATH) / appname

        warnings.warn(
            "Could not find APPDATA environment variable to get user cache dir,"
            " but detected platform 'win32'."
            f" Defaulting to a path '{use_instead_path}'."
            " If you would prefer, please specify a directory when creating"
            " the model.",
            UserWarning,
            stacklevel=2,
        )
        return use_instead_path

    if platform == "darwin":
        return Path.home() / "Library" / "Caches" / appname

    # TODO: Not entirely sure here, Python doesn't explicitly list
    # all of these and defaults to the underlying operating system
    # if not sure.
    linux_likes = ("freebsd", "linux", "netbsd", "openbsd")
    if any(platform.startswith(linux) for linux in linux_likes):
        # The reason to use "" as default is that the env var could exist but be empty.
        # We catch all this with the `.strip() != ""` below
        XDG_CACHE_HOME = os.environ.get("XDG_CACHE_HOME", "")
        if XDG_CACHE_HOME.strip() != "":
            return Path(XDG_CACHE_HOME) / appname
        return Path.home() / ".cache" / appname

    warnings.warn(
        f"Unknown platform '{platform}' to get user cache dir."
        f" Defaulting to a path at the execution site '{use_instead_path}'."
        " If you would prefer, please specify a directory when creating"
        " the model.",
        UserWarning,
        stacklevel=2,
    )
    return use_instead_path
