from __future__ import annotations

import math
import os
import shutil
import sys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.core.models import AbstractModel
from autogluon.features.generators import LabelEncoderFeatureGenerator

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


class TabDPTModel(AbstractModel):
    ag_key = "TABDPT"
    ag_name = "TabDPT"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator = None
        self._predict_hps = None

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_cpus: int = 1,
        num_gpus: int = 0,
        **kwargs,
    ):
        from torch.cuda import is_available

        device = "cuda" if num_gpus != 0 else "cpu"
        if (device == "cuda") and (not is_available()):
            # FIXME: warn instead and switch to CPU.
            raise AssertionError(
                "Fit specified to use GPU, but CUDA is not available on this machine. "
                "Please switch to CPU usage instead.",
            )
        from tabdpt import TabDPTClassifier, TabDPTRegressor

        model_cls = TabDPTClassifier if self.problem_type in [BINARY, MULTICLASS] else TabDPTRegressor

        hps = self._get_model_params()
        self._predict_hps = dict(seed=42, context_size=1024)
        X = self.preprocess(X)
        y = y.to_numpy()
        self.model = model_cls(
            path=self._download_and_get_model_path(),
            device=device,
            use_flash=self._use_flash(),
            **hps,
        )
        self.model.fit(X=X, y=y)

    @staticmethod
    def _use_flash() -> bool:
        """Detect if torch's native flash attention is available on the current machine."""
        import torch

        if not torch.cuda.is_available():
            return False

        device = torch.device("cuda:0")
        capability = torch.cuda.get_device_capability(device)

        if capability == (7, 5):
            return False

        return True

    @staticmethod
    def _download_and_get_model_path() -> str:
        # We follow TabPFN-logic for model caching as /tmp is not a persistent cache location.
        from tabdpt.estimator import TabDPTEstimator
        from tabdpt.utils import download_model

        model_dir = _user_cache_dir(platform=sys.platform, appname="tabdpt")
        model_dir.mkdir(exist_ok=True, parents=True)

        final_model_path = model_dir / Path(TabDPTEstimator._DEFAULT_CHECKPOINT_PATH).name

        if not final_model_path.exists():
            model_path = Path(download_model())  # downloads to /tmp
            shutil.copy(model_path, final_model_path)  # copy to user cache dir

        return str(final_model_path)

    def _get_default_resources(self) -> tuple[int, int]:
        # Use only physical cores for better performance based on benchmarks
        num_cpus = ResourceManager.get_cpu_count(only_physical_cores=True)

        num_gpus = min(1, ResourceManager.get_gpu_count_torch(cuda_only=True))

        return num_cpus, num_gpus

    def get_minimum_resources(self, is_gpu_available: bool = False) -> dict[str, int | float]:
        return {
            "num_cpus": 1,
            "num_gpus": 1 if is_gpu_available else 0,
        }

    def _predict_proba(self, X, **kwargs) -> np.ndarray:
        X = self.preprocess(X, **kwargs)

        # Fix bug in TabDPt where batches of length 1 crash the prediction.
        # - We set the inference size such that there are no batches of length 1.
        math.ceil(len(X) / self.model.inf_batch_size)
        last_batch_size = len(X) % self.model.inf_batch_size
        if last_batch_size == 1:
            self.model.inf_batch_size += 1

        if self.problem_type in [REGRESSION]:
            return self.model.predict(X, **self._predict_hps)

        y_pred_proba = self.model.predict_proba(X, **self._predict_hps)
        return self._convert_proba_to_unified_form(y_pred_proba)

    def _preprocess(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """TabDPT requires numpy array as input."""
        X = super()._preprocess(X, **kwargs)
        if self._feature_generator is None:
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)
        if self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)
        return X.to_numpy()

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["binary", "multiclass", "regression"]

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}

    @classmethod
    def _get_default_ag_args_ensemble(cls, **kwargs) -> dict:
        default_ag_args_ensemble = super()._get_default_ag_args_ensemble(**kwargs)
        extra_ag_args_ensemble = {
            "fold_fitting_strategy": "sequential_local",
            "refit_folds": True,
        }
        default_ag_args_ensemble.update(extra_ag_args_ensemble)
        return default_ag_args_ensemble

# Vendored from TabPFNv2 Code
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

