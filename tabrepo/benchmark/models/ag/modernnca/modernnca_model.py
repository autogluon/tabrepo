from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
from types import SimpleNamespace
from typing import Literal, Optional, List, Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.models import AbstractModel
from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer
from sklearn.utils.validation import check_is_fitted

from tabrepo.benchmark.models.ag.modernnca.modernNCA import ModernNCA
from tabrepo.benchmark.models.ag.modernnca.modernnca_method import ModernNCAMethod

logger = logging.getLogger(__name__)

import math
import random
from pathlib import Path

import scipy
import sklearn
import torch
import numpy as np
from torch import nn


class TabMOrdinalEncoder(BaseEstimator, TransformerMixin):
    # encodes missing and unknown values to a value one larger than the known values
    def __init__(self):
        # No fitted attributes here — only parameters
        pass

    def fit(self, X, y=None):
        X = pd.DataFrame(X)

        # Fit internal OrdinalEncoder with NaNs preserved for now
        self.encoder_ = OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=np.nan,
            encoded_missing_value=np.nan
        )
        self.encoder_.fit(X)

        # Cardinalities = number of known categories per column
        self.cardinalities_ = [len(cats) for cats in self.encoder_.categories_]

        return self

    def transform(self, X):
        check_is_fitted(self, ['encoder_', 'cardinalities_'])

        X = pd.DataFrame(X)
        X_enc = self.encoder_.transform(X)

        # Replace np.nan (unknown or missing) with cardinality value
        for col_idx, cardinality in enumerate(self.cardinalities_):
            mask = np.isnan(X_enc[:, col_idx])
            X_enc[mask, col_idx] = cardinality

        return X_enc.astype(int)

    def get_cardinalities(self):
        check_is_fitted(self, ['cardinalities_'])
        return self.cardinalities_


class ModernNCAImplementation:
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

        problem_type = self.config['problem_type']
        task_type: TaskType = 'binclass' if problem_type == 'binary' else problem_type

        print(f'{task_type=}')

        # hyperparams
        # defaults taken from https://github.com/LAMDA-Tabular/TALENT/blob/cb6cb0cc9d69ac75c467e8dae8ca5ac3d3beb2f2/TALENT/configs/default/modernNCA.json
        num_emb_type = self.config.get('num_emb_type', 'plr')
        num_emb_n_frequencies = self.config.get('num_emb_n_frequencies', 77)
        num_emb_frequency_scale = self.config.get('num_emb_frequency_scale', 0.04431360576139521)
        num_emb_d_embedding = self.config.get('num_emb_d_embedding', 34)
        num_emb_lite = self.config.get('num_emb_lite', True)

        dim = self.config.get('dim', 128)
        sample_rate = self.config.get('sample_rate', 0.5)
        n_epochs = self.config.get('n_epochs', 200)
        batch_size = self.config.get('batch_size', 1024)
        lr = self.config.get('lr', 2e-3)
        d_block = self.config.get('d_block', 512)
        dropout = self.config.get('dropout', 0.1)
        n_blocks = self.config.get('n_blocks', 0)
        weight_decay = self.config.get('weight_decay', 2e-4)
        temperature = self.config.get('temperature', 1.0)

        ds_parts = dict()

        n_classes = None

        self.cat_col_names_ = cat_col_names
        # todo: do we need to do this here or is it already done before?
        self.ord_enc_ = OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=np.nan,
            encoded_missing_value=np.nan
        )
        self.ord_enc_.fit(X_train[cat_col_names])

        for part, X, y in [('train', X_train, y_train), ('val', X_val, y_val)]:
            tensors = dict()

            tensors['x_cat'] = self.ord_enc_.transform(X[cat_col_names])
            x_cont_np = X.drop(columns=cat_col_names).to_numpy(dtype=np.float32)

            tensors['x_cont'] = x_cont_np
            if task_type == 'regression':
                tensors['y'] = y.to_numpy(np.float32)
                if part == 'train':
                    n_classes = 0
            else:
                # todo: we assume that it's already ordinally encoded
                tensors['y'] = y.to_numpy(np.int32)
                if part == 'train':
                    n_classes = tensors['y'].max().item() + 1

            ds_parts[part] = tensors

        self.n_train_ = len(X_train)
        device = self.config['device']
        device = torch.device(device)

        self.n_classes_ = n_classes
        self.task_type_ = task_type
        self.device_ = device

        self.save_path_ = tempfile.mkdtemp()  # todo: can we get a temp folder from outside? such that we can delete it afterwards?
        # print(f'{self.save_path_=}')
        # os.makedirs(self.save_path_)

        config = {
            'training': {
                'lr': lr,
                'weight_decay': weight_decay,
            },
            'model': {
                'dim': dim,
                'dropout': dropout,
                'd_block': d_block,
                'n_blocks': n_blocks,
                'temperature': temperature,
                'sample_rate': sample_rate,
                'num_embeddings': {
                    'type': 'PLREmbeddings',
                    'n_frequencies': num_emb_n_frequencies,
                    'frequency_scale': num_emb_frequency_scale,
                    'd_embedding': num_emb_d_embedding,
                    'lite': num_emb_lite,
                }
            },
        }

        args = {
            'cat_policy': 'tabr_ohe',  # mandatory
            'num_policy': 'none',  # mandatory
            'use_float': True,
            'device': device,
            'num_nan_policy': self.config.get('num_nan_policy', 'mean'),  # todo
            'cat_nan_policy': self.config.get('cat_nan_policy', 'new'),  # todo
            'normalization': self.config.get('normalization', 'standard'),  # todo
            'seed': seed,
            'batch_size': batch_size,
            'max_epoch': n_epochs,
            'save_path': self.save_path_,
            'config': config,
        }
        args = SimpleNamespace(**args)  # convert to an object with the keys as attributes
        info = {
            'task_type': task_type,
            'n_num_features': ds_parts['train']['x_cont'].shape[1],
            'n_cat_features': ds_parts['train']['x_cat'].shape[1],
        }

        # # set empty tensors to None
        # for tensors in ds_parts.values():
        #     for tens_name in ['x_cat', 'x_cont']:
        #         if tensors[tens_name].size == 0:
        #             tensors[tens_name] = None

        data = [{part: ds_parts[part][tens_name] for part in ['train', 'val']} for tens_name in ['x_cont', 'x_cat', 'y']]

        if info['n_num_features'] == 0:
            data[0] = None
        if info['n_cat_features'] == 0:
            data[1] = None

        data = tuple(data)

        assert num_emb_type == 'plr'  # ohthers are currently not implemented

        # todo: time limit
        # todo: custom validation metric
        # todo: is GPU handled correctly?

        # target standardization for regression is automatically done in "data_label_process"
        self.model_ = ModernNCAMethod(args=args, is_regression=task_type == 'regression')
        self.model_.fit(data=data, info=info, train=True, config=config)

    def predict_raw(self, X: pd.DataFrame) -> torch.Tensor:
        tensors = dict()
        tensors['x_cat'] = self.ord_enc_.transform(X[self.cat_col_names_])
        tensors['x_cont'] = X.drop(columns=X[self.cat_col_names_]).to_numpy(dtype=np.float32)

        # for tens_name in ['x_cat', 'x_cont']:
        #     if tensors[tens_name].size == 0:
        #         tensors[tens_name] = None

        # generate dummy y
        if self.task_type_ == 'regression':
            tensors['y'] = np.zeros(tensors['x_cat'].shape[0])
        else:
            tensors['y'] = np.zeros(tensors['x_cat'].shape[0], dtype=np.int32)

        data = [{'test': tensors[tens_name]} for tens_name in ['x_cont', 'x_cat', 'y']]

        # eliminate empty tensors
        for i in range(2):
            if data[i]['test'].size == 0:
                data[i] = None
        data = tuple(data)

        # could use 'epoch-last' instead of 'best-val' for the last model
        y_pred = self.model_.predict(data=data, info=None, model_name='best-val')

        print(f'predict: {X=}, {y_pred=}')

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

    def __del__(self):
        # todo: okay?
        # need the check perhaps because the delete can be called multiple times
        # if the object is serialized, deleted, loaded again, deleted again
        if os.path.exists(self.save_path_):
            shutil.rmtree(self.save_path_)


class ModernNCAModel(AbstractModel):
    ag_key = "MNCA"
    ag_name = "ModernNCA"

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
                       f"WARNING: GPUs are not yet implemented for ModernNCA model, but `num_gpus={num_gpus}` was specified... Ignoring GPU.")

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
        self.model = ModernNCAImplementation(
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
