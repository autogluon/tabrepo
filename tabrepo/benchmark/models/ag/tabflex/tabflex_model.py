from __future__ import annotations

import os
import sys
import types
import warnings
from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING

from autogluon.core.models import AbstractModel
from autogluon.features.generators import LabelEncoderFeatureGenerator

if TYPE_CHECKING:
    import pandas as pd


# FIXME: the below dependencies are not needed and not a required dependency for TabFlex
#  but is not lazy imported in the TabFlex code base. Thus, we monkey patch it to avoid
#  installing it.
sys.modules["mlflow"] = types.ModuleType("mlflow")
fake_gyptorch = types.ModuleType("gpytorch")
fake_gyptorch.models = types.ModuleType("gpytorch.models")
fake_gyptorch.models.ExactGP = ABC
sys.modules["gpytorch"] = fake_gyptorch


def _elu_activation(x):
    import torch

    return torch.nn.functional.elu(x) + 1


def _monkey_patch_ticl_lambda():
    # Monkey patch to avoid lambda pickle error in linear attention code.
    from ticl.models import linear_attention

    linear_attention.elu_feature_map = (
        linear_attention.ActivationFunctionFeatureMap.factory(_elu_activation)
    )


class TabFlex:
    def __init__(
        self,
        *,
        base_path: str | Path,
        tabflexh1k: str,
        tabflexl100: str,
        tabflexs100: str,
        device: str,
        random_state: int,
    ):
        _monkey_patch_ticl_lambda()

        import torch
        from ticl.prediction.tabpfn import TabPFNClassifier

        self.base_path = base_path
        self.tabflexh1k = tabflexh1k
        self.tabflexl100 = tabflexl100
        self.tabflexs100 = tabflexs100
        self.device = device
        self.random_state = random_state

        torch.set_num_threads(1)

        tabflexh1k_model_string = self.tabflexh1k.split("_epoch_")[0]
        tabflexh1k_epoch = self.tabflexh1k.split("_epoch_")[1].split(".cpkt")[0]
        tabflexl100_model_string = self.tabflexl100.split("_epoch_")[0]
        tabflexl100_epoch = self.tabflexl100.split("_epoch_")[1].split(".cpkt")[0]
        tabflexs100_model_string = self.tabflexs100.split("_epoch_")[0]
        tabflexs100_epoch = self.tabflexs100.split("_epoch_")[1].split(".cpkt")[0]

        # All hardcoded values are from the TabFlex Code
        shared_kwargs = {
            "base_path": self.base_path,
            "device": self.device,
            "seed": self.random_state,
        }
        self.tabflexh1k = TabPFNClassifier(
            model_string=tabflexh1k_model_string,
            N_ensemble_configurations=3,
            epoch=tabflexh1k_epoch,
            **shared_kwargs,
        )
        self.tabflexl100 = TabPFNClassifier(
            model_string=tabflexl100_model_string,
            N_ensemble_configurations=1,
            epoch=tabflexl100_epoch,
            **shared_kwargs,
        )

        self.tabflexs100 = TabPFNClassifier(
            model_string=tabflexs100_model_string,
            N_ensemble_configurations=3,
            epoch=tabflexs100_epoch,
            **shared_kwargs,
        )

    # FIXME: for refit it can happen that the model changes due to different sample
    #  size withotu CV, which would be very bad for validation / HPO.
    def fit(self, X, y):
        N, D = X.shape

        if N >= 3000 and D <= 100:
            self.model = self.tabflexl100
        elif D > 100 or (D / N >= 0.2 and N >= 3000):
            if D <= 1000:
                self.model = self.tabflexh1k
            else:
                self.model = self.tabflexh1k
                self.model.dimension_reduction = "random_proj"
                self.model.fit(X, y, overwrite_warning=True)
                return self
        else:
            self.model = self.tabflexs100

        self.model.fit(X, y, overwrite_warning=True)

        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


# TODO: Needs memory usage estimate method
class TabFlexModel(AbstractModel):
    ag_key = "TABFLEX"
    ag_name = "TabFlex"

    # TabFlex Hardcoded model names
    tabflexh1k = "ssm_tabpfn_b4_maxnumclasses100_modellinear_attention_numfeatures1000_n1024_validdatanew_warm_08_23_2024_19_25_40_epoch_3140.cpkt"
    tabflexl100 = "ssm_tabpfn_b4_largedatasetTrue_modellinear_attention_nsamples50000_08_01_2024_22_05_50_epoch_110.cpkt"
    tabflexs100 = "ssm_tabpfn_modellinear_attention_08_28_2024_19_00_44_epoch_3110.cpkt"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cat_col_names_ = None

    def get_model_cls(self):
        if self.problem_type not in ["binary", "multiclass"]:
            raise AssertionError(f"Unsupported problem_type: {self.problem_type}")

        return TabFlex

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_gpus: int = 0,
        **kwargs,
    ):
        device = self._get_device(num_gpus)
        hyp = self._get_model_params()

        base_path = self._download_all_models()
        self.model = self.get_model_cls()(
            base_path=base_path,
            tabflexh1k=self.tabflexh1k,
            tabflexl100=self.tabflexl100,
            tabflexs100=self.tabflexs100,
            device=device,
            **hyp,
        )

        X = self.preprocess(X, is_train=True)
        self.model = self.model.fit(
            X=X,
            y=y,
        )

    def _set_default_params(self):
        default_params = {
            "random_state": 42,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["binary", "multiclass"]

    def _get_default_resources(self) -> tuple[int, int]:
        import torch
        from autogluon.common.utils.resource_utils import ResourceManager

        num_cpus = ResourceManager.get_cpu_count_psutil()
        num_gpus = 1 if torch.cuda.is_available() else 0
        return num_cpus, num_gpus

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}

    def _get_device(self, num_gpus: int) -> str:
        import torch

        device = "cuda" if num_gpus != 0 else "cpu"
        if (device == "cuda") and (not torch.cuda.is_available()):
            # FIXME: warn instead and switch to CPU.
            raise AssertionError(
                "Fit specified to use GPU, but CUDA is not available on this machine. "
                "Please switch to CPU usage instead.",
            )
        return device

    # FIXME: clarify how to handle categorical features as they are not passed
    #  to the model, but are encoded as ordinal features only.
    def _preprocess(
        self,
        X: pd.DataFrame,
        is_train: bool = False,
        bool_to_cat: bool = False,
        impute_bool: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        X = super()._preprocess(X, **kwargs)

        # Ordinal Encoding of cat features but keep as cat
        if is_train:
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)
        if self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(
                X=X
            )
            if self.cat_col_names_ is None:
                self.cat_col_names_ = self._feature_generator.features_in[:]
        else:
            self.cat_col_names_ = []

        return X

    @staticmethod
    def _download_all_models() -> str:
        # We follow TabPFN-logic for model caching as /tmp is not a persistent cache
        # location.
        import urllib

        _monkey_patch_ticl_lambda()
        from ticl.utils import DownloadProgressBar

        base_model_dir = _user_cache_dir(
            platform=sys.platform, appname="tabflex"
        ).resolve()
        model_dir = base_model_dir / "models_diff"
        model_dir.mkdir(exist_ok=True, parents=True)

        for model_name in [
            TabFlexModel.tabflexh1k,
            TabFlexModel.tabflexl100,
            TabFlexModel.tabflexs100,
        ]:
            final_model_path = (model_dir / model_name).resolve()

            if not final_model_path.exists():
                url = f"https://amuellermothernet.blob.core.windows.net/models/{model_name}"
                print(
                    f"Downloading model from {url} to {final_model_path}. This can take a bit."
                )
                with DownloadProgressBar(
                    unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
                ) as t:
                    urllib.request.urlretrieve(
                        url, filename=final_model_path, reporthook=t.update_to
                    )

        return str(base_model_dir)


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
