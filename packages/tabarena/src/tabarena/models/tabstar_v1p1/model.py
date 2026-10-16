from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.models.abstract import SharedWeights
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

if TYPE_CHECKING:
    import pandas as pd


logger = logging.getLogger(__name__)

#: Commits pinned so the checkpoints fetched here never silently change if a repo's default branch
#: moves. Bump deliberately (with a note on what changed) when picking up a newer checkpoint.
_HF_REVISION_CLS = "4480a25fea83029b02b07c10b9f03a376481daaf"
_HF_REVISION_REG = "076160ddcfe71faa98841df274bc0ebdc5d2c093"
#: The text encoder whose tokenizer the library loads by name alongside the weights.
_HF_TOKENIZER_REPO = "intfloat/e5-small-v2"
_HF_TOKENIZER_REVISION = "ffb93f3bd4047442299a41ebb6fa998a38507c52"

_HF_CHECKPOINTS = {
    True: ("alana89/TabSTAR-v1.1-cls-20261011", _HF_REVISION_CLS),
    False: ("alana89/TabSTAR-v1.1-reg-20261009", _HF_REVISION_REG),
}


def _pinned_checkpoint_path(is_cls: bool) -> str:
    """Local snapshot dir of the pinned checkpoint, downloading it if the cache is cold."""
    from huggingface_hub import snapshot_download

    repo_id, revision = _HF_CHECKPOINTS[is_cls]
    return snapshot_download(repo_id=repo_id, revision=revision)


class TabSTARv1p1Model(AbstractTorchModel):
    """TabSTAR v1.1: https://arxiv.org/abs/2505.18125.

    Second generation of TabSTAR, pretrained on a larger corpus and shipped as a separate estimator
    in the same ``tabstar`` package. Classification and regression load different checkpoints.
    """

    ag_key = "TA-TABSTAR-V1.1"
    ag_name = "TA-TabSTAR-v1.1"
    ag_priority = 65
    seed_name = "random_state"
    _supported_problem_types = ["binary", "multiclass", "regression"]
    default_num_gpus = 1
    default_resources_physical_cores_only = True
    minimum_num_gpus = 1
    _default_ag_args_ensemble_extra: ClassVar[dict] = {
        # Parallel folding crashes if the weights aren't pre-downloaded.
        # TODO: switch to parallel fitting on one GPU once VRAM memory estimation is supported.
        "fold_fitting_strategy": "sequential_local",
        "refit_folds": True,
    }
    warmup_modules: ClassVar[tuple[str, ...]] = (
        "torch",
        "tabstar.tabstarv1p1_model",
        "tabstar.tabstarv1p1.finetune.hyperparams",
    )
    #: The library builds the base model through ``TabSTARModel.from_pretrained`` twice per fit (to
    #: train, and to attach the saved adapters). The checkpoint is read once per process; every
    #: fit fine-tunes its own deep copy and pickles it.
    shared_weights: ClassVar[SharedWeights] = SharedWeights(
        loader="tabstar.tabstarv1p1.arch.model:TabSTARModel.from_pretrained",
        key=("pretrained_model_name_or_path",),
        copy_per_fit=True,
    )
    #: Knobs that make the warm-up's dummy fit cheap without touching the network.
    cheap_hyperparameters: ClassVar[dict] = {"max_epochs": 1}

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        time_limit: float | None = None,
        num_gpus: int = 0,
        **kwargs,
    ):
        import torch

        available_num_gpus = ResourceManager.get_gpu_count_torch(cuda_only=True)
        if num_gpus > available_num_gpus:
            raise AssertionError(
                f"Fit specified to use {num_gpus} GPU, but only {available_num_gpus} "
                "CUDA GPUs are available. Please activate CUDA or switch to CPU usage.",
            )
        device = "cuda" if num_gpus != 0 else "cpu"
        if (device == "cuda") and (not torch.cuda.is_available()):
            raise AssertionError(
                "Fit specified to use GPU, but CUDA is not available on this machine. "
                "Please switch to CPU usage instead.",
            )

        from tabstar.tabstarv1p1.finetune.hyperparams import LORA_BATCH, VAL_BATCH
        from tabstar.tabstarv1p1_model import TabSTARv1p1Classifier, TabSTARv1p1Regressor

        if self.problem_type in ["binary", "multiclass"]:
            model_cls = TabSTARv1p1Classifier
            is_cls = True
        elif self.problem_type in ["regression"]:
            model_cls = TabSTARv1p1Regressor
            is_cls = False
        else:
            raise AssertionError(f"Unsupported problem_type: {self.problem_type}")

        # Simple heuristic for batch size
        train_batch_size = LORA_BATCH
        predict_batch_size = VAL_BATCH
        if X.shape[1] > 200:
            if X.shape[0] > 50_000:
                train_batch_size = 16
                predict_batch_size = 16
            else:
                train_batch_size = 64
                predict_batch_size = 64

        hps = self._get_model_params()
        self.model = model_cls(
            **hps,
            lora_batch=train_batch_size,
            val_batch_size=predict_batch_size,
            time_limit=time_limit,
            device=device,
            output_dir=self.path + "/model_checkpoints",
            # The library resolves a checkpoint by repo id, which always tracks the default branch.
            # Hand it the pinned snapshot directory instead so the fit is reproducible.
            pretrain_dataset_or_path=_pinned_checkpoint_path(is_cls=is_cls),
        )

        if X_val is None:
            from autogluon.core.utils import generate_train_test_split

            X, X_val, y, y_val = generate_train_test_split(
                X=X,
                y=y,
                problem_type=self.problem_type,
                test_size=0.33,
                random_state=0,
            )

        # Does nothing but might be used for future extensions
        X = self.preprocess(X, y=y)
        if X_val is not None:
            X_val = self.preprocess(X_val)
        # Inverse label transformation to retain original semantics for classification
        if self.problem_type in ["binary", "multiclass"]:
            if (not hasattr(self, "label_cleaner")) or (self.label_cleaner is None):
                raise ValueError("Label cleaner missing from AbstractModel!")

            y = self.label_cleaner.inverse_transform(y)
            if y_val is not None:
                y_val = self.label_cleaner.inverse_transform(y_val)

        self.model.fit(
            X=X,
            y=y,
            x_val=X_val,
            y_val=y_val,
        )

    # ``max_epochs`` is deliberately left at the library default: v1.1 builds its OneCycle schedule
    # over ``steps_per_epoch * max_epochs``, so inflating it to stop on time limit or early stopping
    # instead would keep every real epoch inside the warm-up ramp and never anneal the LR.

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}

    @classmethod
    def _class_tags(cls):
        # The estimator places its model itself; AutoGluon moves nothing on save or load.
        return {
            **super()._class_tags(),
            "can_set_device": False,
            "set_device_on_save_to": None,
            "set_device_on_load": False,
        }


def prefetch_weights() -> None:
    """Pre-download both pinned checkpoints and the tokenizer (warms the cache for offline fits)."""
    from transformers import AutoTokenizer

    for is_cls in (True, False):
        _pinned_checkpoint_path(is_cls=is_cls)
    AutoTokenizer.from_pretrained(_HF_TOKENIZER_REPO, revision=_HF_TOKENIZER_REVISION)
