from __future__ import annotations

import logging
import time
from typing import Literal, Optional, List, Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.models import AbstractModel
from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer
from sklearn.utils.validation import check_is_fitted

from tabrepo.benchmark.models.ag.tabm import rtdl_num_embeddings, tabm_reference

logger = logging.getLogger(__name__)

import math
import random
from pathlib import Path

import scipy
import sklearn
import torch
import numpy as np
from torch import nn


# partially adapted from pytabkit's TabM implementation


def get_tabm_auto_batch_size(n_train: int) -> int:
    # taken from tabr paper table 14
    # the cutoffs might not be exactly the same
    if n_train < 10_000:
        return 128
    elif n_train < 30_000:
        return 256
    elif n_train < 200_000:
        return 512
    else:
        return 1024


class RTDLQuantileTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, noise=1e-5, random_state=None, n_quantiles=1000, subsample=1_000_000_000,
                 output_distribution="normal"):
        self.noise = noise
        self.random_state = random_state
        self.n_quantiles = n_quantiles
        self.subsample = subsample
        self.output_distribution = output_distribution

    def fit(self, X, y=None):
        # Calculate the number of quantiles based on data size
        n_quantiles = max(min(X.shape[0] // 30, self.n_quantiles), 10)

        # Initialize QuantileTransformer
        normalizer = QuantileTransformer(
            output_distribution=self.output_distribution,
            n_quantiles=n_quantiles,
            subsample=self.subsample,
            random_state=self.random_state
        )

        # Add noise if required
        X_modified = self._add_noise(X) if self.noise > 0 else X

        # Fit the normalizer
        normalizer.fit(X_modified)
        # show that it's fitted
        self.normalizer_ = normalizer

        return self

    def transform(self, X, y=None):
        check_is_fitted(self)
        return self.normalizer_.transform(X)

    def _add_noise(self, X):
        # todo: small adaptation, adding noise on a scale relative to the feature's standard deviation
        #  instead of just using an absolute scale for the noise
        stds = np.std(X, axis=0, keepdims=True)
        rng = np.random.default_rng(self.random_state)
        return X + self.noise * stds * rng.standard_normal(X.shape)


class TabMImplementation:
    def __init__(self, **config):
        self.config = config

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series,
            cat_col_names: List[Any], time_to_fit_in_seconds: Optional[float] = None):
        seed: Optional[int] = self.config.get('random_state', None)
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        start_time = time.time()

        if 'n_threads' in self.config:
            torch.set_num_threads(self.config['n_threads'])

        if X_val is None or len(X_val) == 0:
            raise ValueError(f'Training without validation set is currently not implemented')

        TaskType = Literal['regression', 'binclass', 'multiclass']

        task_type: TaskType = self.config['problem_type']

        # hyperparams
        # set defaults to tabm-mini defaults
        arch_type = self.config.get('arch_type', 'tabm-mini')
        num_emb_type = self.config.get('num_emb_type', 'pwl')
        n_epochs = self.config.get('n_epochs', 1_000_000_000)
        patience = self.config.get('patience', 16)
        batch_size = self.config.get('batch_size', 'auto')
        compile_model = self.config.get('compile_model', False)
        lr = self.config.get('lr', 2e-3)
        d_embedding = self.config.get('d_embedding', 16)
        d_block = self.config.get('d_block', 512)
        dropout = self.config.get('dropout', 0.1)
        tabm_k = self.config.get('tabm_k', 32)
        allow_amp = self.config.get('allow_amp', False)
        n_blocks = self.config.get('n_blocks', 'auto')
        num_emb_n_bins = self.config.get('num_emb_n_bins', 48)
        eval_batch_size = self.config.get('eval_batch_size', 1024)
        share_training_batches = self.config.get('share_training_batches', False)

        # todo: is this okay? otherwise the test fails
        num_emb_n_bins = min(num_emb_n_bins, len(X_train)-1)
        if len(X_train) <= 2:
            num_emb_type = 'none'  # there is no valid number of bins for piecewise linear embeddings

        if batch_size == 'auto':
            batch_size = get_tabm_auto_batch_size(n_train=len(X_train))

        weight_decay = self.config.get('weight_decay', 0.0)
        gradient_clipping_norm = self.config.get('gradient_clipping_norm', 1.0)  # this is the search space default

        ds_parts = dict()

        n_classes = None

        self.cat_col_names_ = cat_col_names
        # todo: do we need to do this here or is it already done before?
        self.ord_enc_ = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1,
                                       encoded_missing_value=-1)
        self.ord_enc_.fit(X_train[cat_col_names])
        self.num_prep_ = RTDLQuantileTransformer()

        for part, X, y in [('train', X_train, y_train), ('val', X_val, y_val)]:
            tensors = dict()

            tensors['x_cat'] = torch.as_tensor(self.ord_enc_.transform(X[cat_col_names]), dtype=torch.long)
            x_cont_np = X.drop(columns=cat_col_names).to_numpy(dtype=np.float32)

            if part == 'train':
                self.num_prep_.fit(x_cont_np)

            tensors['x_cont'] = torch.as_tensor(self.num_prep_.transform(x_cont_np))
            if task_type == 'regression':
                tensors['y'] = torch.as_tensor(y.to_numpy(np.float32))
                if part == 'train':
                    n_classes = 0
            else:
                # todo: we assume that it's already ordinally encoded
                tensors['y'] = torch.as_tensor(y.to_numpy(np.int32), dtype=torch.long)
                if part == 'train':
                    n_classes = tensors['y'].max().item() + 1

            ds_parts[part] = tensors

        n_train = len(X_train)
        cat_cardinalities = ds_parts['train']['x_cat'].max(dim=0)[0].numpy().tolist()
        device = self.config['device']
        device = torch.device(device)

        self.n_classes_ = n_classes
        self.task_type_ = task_type
        self.device_ = device

        part_names = ['train', 'val']

        # filter out numerical columns with only a single value
        x_cont_train = ds_parts['train']['x_cont']

        # todo: do we need to do this or does AG do it already?
        # mask of which columns are not constant
        self.num_col_mask_ = ~torch.all(x_cont_train == x_cont_train[0:1, :], dim=0)

        for part in part_names:
            ds_parts[part]['x_cont'] = ds_parts[part]['x_cont'][:, self.num_col_mask_]
            # tensor infos are not correct anymore, but might not be used either

        for part in part_names:
            for tens_name in ds_parts[part]:
                ds_parts[part][tens_name] = ds_parts[part][tens_name].to(device)

        # update
        n_cont_features = ds_parts['train']['x_cont'].shape[1]

        Y_train = ds_parts['train']['y'].clone()
        if task_type == 'regression':
            self.y_mean_ = ds_parts['train']['y'].mean().item()
            self.y_std_ = ds_parts['train']['y'].std(correction=0).item()

            Y_train = (Y_train - self.y_mean_) / (self.y_std_ + 1e-30)

        # the | operator joins dicts (like update() but not in-place)
        data = {part: dict(x_cont=ds_parts[part]['x_cont'], y=ds_parts[part]['y']) | (
            dict(x_cat=ds_parts[part]['x_cat']) if ds_parts[part]['x_cat'].shape[1] > 0 else dict())
                for part in part_names}

        # adapted from https://github.com/yandex-research/tabm/blob/main/example.ipynb

        # Automatic mixed precision (AMP)
        # torch.float16 is implemented for completeness,
        # but it was not tested in the project,
        # so torch.bfloat16 is used by default.
        amp_dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
            if torch.cuda.is_available()
            else None
        )
        # Changing False to True will result in faster training on compatible hardware.
        amp_enabled = allow_amp and amp_dtype is not None
        grad_scaler = torch.cuda.amp.GradScaler() if amp_dtype is torch.float16 else None  # type: ignore

        # fmt: off
        logger.info(f'Device:        {device.type.upper()}'
                    f'\nAMP:           {amp_enabled} (dtype: {amp_dtype})'
                    f'\ntorch.compile: {compile_model}'
                    )
        # fmt: on

        bins = None if num_emb_type != 'pwl' or n_cont_features == 0 else rtdl_num_embeddings.compute_bins(
            data['train']['x_cont'], n_bins=num_emb_n_bins)

        model = tabm_reference.Model(
            n_num_features=n_cont_features,
            cat_cardinalities=cat_cardinalities,
            n_classes=n_classes if n_classes > 0 else None,
            backbone={
                'type': 'MLP',
                'n_blocks': n_blocks if n_blocks != 'auto' else (3 if bins is None else 2),
                'd_block': d_block,
                'dropout': dropout,
            },
            bins=bins,
            num_embeddings=(
                None
                if bins is None
                else {
                    'type': 'PiecewiseLinearEmbeddings',
                    'd_embedding': d_embedding,
                    'activation': False,
                    'version': 'B',
                }
            ),
            arch_type=arch_type,
            k=tabm_k,
            share_training_batches=share_training_batches,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        if compile_model:
            # NOTE
            # `torch.compile` is intentionally called without the `mode` argument
            # (mode="reduce-overhead" caused issues during training with torch==2.0.1).
            model = torch.compile(model)
            evaluation_mode = torch.no_grad
        else:
            evaluation_mode = torch.inference_mode

        @torch.autocast(device.type, enabled=amp_enabled, dtype=amp_dtype)  # type: ignore[code]
        def apply_model(part: str, idx: torch.Tensor) -> torch.Tensor:
            return (
                model(
                    data[part]['x_cont'][idx],
                    data[part]['x_cat'][idx] if 'x_cat' in data[part] else None,
                )
                .squeeze(-1)  # Remove the last dimension for regression tasks.
                .float()
            )

        base_loss_fn = torch.nn.functional.mse_loss if task_type == 'regression' else torch.nn.functional.cross_entropy

        def loss_fn(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
            # TabM produces k predictions per object. Each of them must be trained separately.
            # (regression)     y_pred.shape == (batch_size, k)
            # (classification) y_pred.shape == (batch_size, k, n_classes)
            k = y_pred.shape[1]
            return base_loss_fn(y_pred.flatten(0, 1),
                                y_true.repeat_interleave(k) if model.share_training_batches else y_true)

        @evaluation_mode()
        def evaluate(part: str) -> float:
            model.eval()

            # When using torch.compile, you may need to reduce the evaluation batch size.
            y_pred: np.ndarray = (
                torch.cat(
                    [
                        apply_model(part, idx)
                        for idx in torch.arange(len(data[part]['y']), device=device).split(
                        eval_batch_size
                    )
                    ]
                )
                .cpu()
                .numpy()
            )
            if task_type == 'regression':
                # Transform the predictions back to the original label space.
                y_pred = y_pred * self.y_std_ + self.y_mean_

            # Compute the mean of the k predictions.
            average_logits = self.config.get('average_logits', False)
            if average_logits:
                y_pred = y_pred.mean(1)
            if task_type != 'regression':
                # For classification, the mean must be computed in the probability space.
                y_pred = scipy.special.softmax(y_pred, axis=-1)
            if not average_logits:
                y_pred = y_pred.mean(1)

            y_true = data[part]['y'].cpu().numpy()
            # todo: here we need more metrics. Can we use ones from AutoGluon?
            score = (
                -(sklearn.metrics.mean_squared_error(y_true, y_pred) ** 0.5)
                if task_type == 'regression'
                else sklearn.metrics.accuracy_score(y_true, y_pred.argmax(1))
            )
            return float(score)  # The higher -- the better.

        # print(f'Test score before training: {evaluate("test"):.4f}')

        epoch_size = math.ceil(n_train / batch_size)
        best = {
            'val': -math.inf,
            # 'test': -math.inf,
            'epoch': -1,
        }
        best_params = [p.clone() for p in model.parameters()]
        # Early stopping: the training stops when
        # there are more than `patience` consequtive bad updates.
        remaining_patience = patience

        try:
            if self.config.get('verbosity', 0) >= 1:
                from tqdm.std import tqdm
            else:
                tqdm = lambda arr, desc: arr
        except ImportError:
            tqdm = lambda arr, desc: arr

        logger.info('-' * 88 + '\n')
        for epoch in range(n_epochs):
            # check time limit
            if epoch > 0 and time_to_fit_in_seconds is not None:
                cur_time = time.time()
                pred_time_after_next_epoch = (epoch + 1) / epoch * (cur_time - start_time)
                if pred_time_after_next_epoch > time_to_fit_in_seconds:
                    break

            batches = (
                torch.randperm(n_train, device=device).split(batch_size)
                if model.share_training_batches
                else [
                    x.transpose(0, 1).flatten()
                    for x in torch.rand((model.k, n_train), device=device)
                    .argsort(dim=1)
                    .split(batch_size, dim=1)
                ]
            )

            for batch_idx in tqdm(batches, desc=f'Epoch {epoch}'):
                model.train()
                optimizer.zero_grad()
                loss = loss_fn(apply_model('train', batch_idx), Y_train[batch_idx])

                # added from https://github.com/yandex-research/tabm/blob/main/bin/model.py
                if gradient_clipping_norm is not None and gradient_clipping_norm != 'none':
                    if grad_scaler is not None:
                        grad_scaler.unscale_(optimizer)
                    nn.utils.clip_grad.clip_grad_norm_(
                        model.parameters(), gradient_clipping_norm
                    )

                if grad_scaler is None:
                    loss.backward()
                    optimizer.step()
                else:
                    grad_scaler.scale(loss).backward()  # type: ignore
                    grad_scaler.step(optimizer)
                    grad_scaler.update()

            val_score = evaluate('val')
            logger.info(f'(val) {val_score:.4f}')

            if val_score > best['val']:
                logger.info('🌸 New best epoch! 🌸')
                # best = {'val': val_score, 'test': test_score, 'epoch': epoch}
                best = {'val': val_score, 'epoch': epoch}
                remaining_patience = patience
                with torch.no_grad():
                    for bp, p in zip(best_params, model.parameters()):
                        bp.copy_(p)
            else:
                remaining_patience -= 1

            if remaining_patience < 0:
                break

            logger.info('')

        logger.info('\n\nResult:')
        logger.info(str(best))

        logger.info(f'Restoring best model')
        with torch.no_grad():
            for bp, p in zip(best_params, model.parameters()):
                p.copy_(bp)

        self.model_ = model

        return None

    def predict_raw(self, X: pd.DataFrame) -> torch.Tensor:
        self.model_.eval()

        tensors = dict()
        tensors['x_cat'] = torch.as_tensor(self.ord_enc_.transform(X[self.cat_col_names_]), dtype=torch.long).to(
            self.device_)
        tensors['x_cont'] = torch.as_tensor(
            self.num_prep_.transform(X.drop(columns=X[self.cat_col_names_]).to_numpy(dtype=np.float32))).to(
            self.device_)

        tensors['x_cont'] = tensors['x_cont'][:, self.num_col_mask_]

        eval_batch_size = self.config.get('eval_batch_size', 1024)
        with torch.no_grad():
            y_pred: torch.Tensor = (
                torch.cat(
                    [
                        self.model_(
                            tensors['x_cont'][idx],
                            tensors['x_cat'][idx] if not tensors['x_cat'].numel() == 0 else None,
                        )
                        .squeeze(-1)  # Remove the last dimension for regression tasks.
                        .float()
                        for idx in torch.arange(tensors['x_cont'].shape[0], device=self.device_).split(
                        eval_batch_size
                    )
                    ]
                )
            )
        if self.task_type_ == 'regression':
            # Transform the predictions back to the original label space.
            y_pred = y_pred * self.y_std_ + self.y_mean_
            y_pred = y_pred.mean(1)
            # y_pred = y_pred.unsqueeze(-1)  # add extra "features" dimension
        else:
            average_logits = self.config.get('average_logits', False)
            if average_logits:
                y_pred = y_pred.mean(1)
            else:
                # For classification, the mean must be computed in the probability space.
                y_pred = torch.log(torch.softmax(y_pred, dim=-1).mean(1) + 1e-30)

        return y_pred.cpu()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        y_pred = self.predict_raw(X)
        if self.task_type_ == 'regression':
            return y_pred.numpy()
        else:
            return y_pred.argmax(dim=-1).numpy()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probas = torch.softmax(self.predict_raw(X), dim=-1).numpy()
        if probas.shape[1] == 2:
            probas = probas[:, 1]
        return probas




# pip install pytabkit
class TabMModel(AbstractModel):
    ag_key = "TABM"
    ag_name = "TabM"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._imputer = None
        self._features_to_impute = None
        self._features_to_keep = None
        self._indicator_columns = None
        self._features_bool = None
        self._bool_to_cat = None

    def _fit(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            X_val: pd.DataFrame = None,
            y_val: pd.Series = None,
            time_limit: float = None,
            num_cpus: int = 1,
            num_gpus: float = 0,
            **kwargs,
    ):
        if num_gpus > 0:
            logger.log(30,
                       f"WARNING: GPUs are not yet implemented for RealMLP model, but `num_gpus={num_gpus}` was specified... Ignoring GPU.")

        hyp = self._get_model_params()

        # todo: not implemented yet
        metric_map = {
            "roc_auc": "1-auc_ovr_alt",
            "accuracy": "class_error",
            "balanced_accuracy": "1-balanced_accuracy",
            "log_loss": "cross_entropy",
            "rmse": "rmse",
            "root_mean_squared_error": "rmse",
            "r2": "rmse",
            "mae": "mae",
            "mean_average_error": "mae",
        }

        val_metric_name = metric_map.get(self.stopping_metric.name, None)

        init_kwargs = dict()

        if val_metric_name is not None:
            init_kwargs["val_metric_name"] = val_metric_name

        if X_val is None:
            # todo
            raise NotImplementedError()

        bool_to_cat = hyp.pop("bool_to_cat", True)
        impute_bool = hyp.pop("impute_bool", True)

        # TODO: GPU
        self.model = TabMImplementation(
            n_threads=num_cpus,
            device="cpu",
            problem_type=self.problem_type,
            **init_kwargs,
            **hyp,
        )

        X = self.preprocess(X, is_train=True, bool_to_cat=bool_to_cat, impute_bool=impute_bool)

        if X_val is not None:
            X_val = self.preprocess(X_val)

        self.model.fit(
            X_train=X,
            y_train=y,
            X_val=X_val,
            y_val=y_val,
            cat_col_names=X.select_dtypes(include='category').columns.tolist(),
            time_to_fit_in_seconds=time_limit,
        )

    # TODO: Move missing indicator + mean fill to a generic preprocess flag available to all models
    # FIXME: bool_to_cat is a hack: Maybe move to abstract model?
    def _preprocess(self, X: pd.DataFrame, is_train: bool = False, bool_to_cat: bool = False, impute_bool: bool = True,
                    **kwargs) -> pd.DataFrame:
        """
        Imputes missing values via the mean and adds indicator columns for numerical features.
        Converts indicator columns to categorical features to avoid them being treated as numerical by RealMLP.
        """
        X = super()._preprocess(X, **kwargs)

        # FIXME: is copy needed?
        X = X.copy(deep=True)
        if is_train:
            self._bool_to_cat = bool_to_cat
            self._features_bool = self._feature_metadata.get_features(required_special_types=["bool"])
            if impute_bool:  # Technically this should do nothing useful because bools will never have NaN
                self._features_to_impute = self._feature_metadata.get_features(valid_raw_types=["int", "float"])
                self._features_to_keep = self._feature_metadata.get_features(invalid_raw_types=["int", "float"])
            else:
                self._features_to_impute = self._feature_metadata.get_features(valid_raw_types=["int", "float"],
                                                                               invalid_special_types=["bool"])
                self._features_to_keep = [f for f in self._feature_metadata.get_features() if
                                          f not in self._features_to_impute]
            if self._features_to_impute:
                self._imputer = SimpleImputer(strategy="mean", add_indicator=True)
                self._imputer.fit(X=X[self._features_to_impute])
                self._indicator_columns = [c for c in self._imputer.get_feature_names_out() if
                                           c not in self._features_to_impute]
        if self._imputer is not None:
            X_impute = self._imputer.transform(X=X[self._features_to_impute])
            X_impute = pd.DataFrame(X_impute, index=X.index, columns=self._imputer.get_feature_names_out())
            if self._indicator_columns:
                # FIXME: Use CategoryFeatureGenerator? Or tell the model which is category
                # TODO: Add to features_bool?
                X_impute[self._indicator_columns] = X_impute[self._indicator_columns].astype("category")
            X = pd.concat([X[self._features_to_keep], X_impute], axis=1)
        if self._bool_to_cat and self._features_bool:
            # FIXME: Use CategoryFeatureGenerator? Or tell the model which is category
            X[self._features_bool] = X[self._features_bool].astype("category")
        return X

    def _set_default_params(self):
        default_params = dict(
            random_state=0,
        )
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["binary", "multiclass", "regression"]

    def _get_default_stopping_metric(self):
        return self.eval_metric

    def _get_default_resources(self) -> tuple[int, int]:
        # logical=False is faster in training
        num_cpus = ResourceManager.get_cpu_count_psutil(logical=False)
        num_gpus = 0  # TODO: Test GPU support
        return num_cpus, num_gpus

    def _estimate_memory_usage(self, X: pd.DataFrame, **kwargs) -> int:
        hyperparameters = self._get_model_params()
        return self.estimate_memory_usage_static(X=X, problem_type=self.problem_type, num_classes=self.num_classes,
                                                 hyperparameters=hyperparameters, **kwargs)

    # FIXME: Find a better estimate for memory usage of TabM. Currently borrowed from FASTAI estimate.
    @classmethod
    def _estimate_memory_usage_static(
            cls,
            *,
            X: pd.DataFrame,
            **kwargs,
    ) -> int:
        return 10 * get_approximate_df_mem_usage(X).sum()

    @classmethod
    def _class_tags(cls):
        return {"can_estimate_memory_usage_static": True}

    def _more_tags(self) -> dict:
        # TODO: Need to add train params support, track best epoch
        #  How to force stopping at a specific epoch?
        tags = {"can_refit_full": False}
        return tags
