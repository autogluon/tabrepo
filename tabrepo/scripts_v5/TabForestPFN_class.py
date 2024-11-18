from __future__ import annotations

import numpy as np
import pandas as pd

from .abstract_class import AbstractExecModel

from pathlib import Path
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels


# TODO: To mitigate val overfitting, can fit multiple random seeds at same time and pick same epoch for all of them, track average performance on epoch.
# TODO: Test shuffling the data and see if it makes TabPFNv2 worse, same with TabForestPFN
class TabForestPFNClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, n_classes, cfg, split_val, path_to_weights):
        from tabularbench.core.trainer_finetune import TrainerFinetune
        from tabularbench.models.foundation.foundation_transformer import FoundationTransformer
        model = FoundationTransformer(
            n_features=cfg.hyperparams['n_features'],
            n_classes=cfg.hyperparams['n_classes'],
            dim=cfg.hyperparams['dim'],
            n_layers=cfg.hyperparams['n_layers'],
            n_heads=cfg.hyperparams['n_heads'],
            attn_dropout=cfg.hyperparams['attn_dropout'],
            y_as_float_embedding=cfg.hyperparams['y_as_float_embedding'],
            use_pretrained_weights=cfg.hyperparams['use_pretrained_weights'],
            path_to_weights=str(Path(path_to_weights)),
        )
        self.split_val = split_val
        self.trainer = TrainerFinetune(cfg, model, n_classes=n_classes)
        super().__init__()

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None):
        from tabularbench.core.dataset_split import make_stratified_dataset_split
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y

        if X_val is not None and y_val is not None:
            X_train, X_valid, y_train, y_valid = X, X_val, y, y_val
        elif self.split_val:
            X_train, X_valid, y_train, y_valid = make_stratified_dataset_split(X, y)
        else:
            X_train, X_valid, y_train, y_valid = X, X, y, y
        self.trainer.train(X_train, y_train, X_valid, y_valid)

        return self

    def predict(self, X):
        logits = self.trainer.predict(self.X_, self.y_, X)
        return logits.argmax(axis=1)

    def predict_proba(self, X):
        logits = self.trainer.predict(self.X_, self.y_, X)
        return np.exp(logits) / np.exp(logits).sum(axis=1)[:, None]


# FIXME: test_epoch might be better if it uses the for loop logic with n_ensembles during finetuning to better estimate val score
# FIXME: Is the model deterministic?
class CustomTabForestPFN(AbstractExecModel):
    def __init__(self, n_ensembles=1, max_epochs=0, split_val=False, path_config=None, path_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.n_ensembles = n_ensembles
        self.max_epochs = max_epochs
        self.split_val = split_val
        if path_config is None:
            path_config = "/home/ubuntu/workspace/code/tabrepo/tabrepo/scripts_v5/ag_models/config_run.yaml"
        self.path_config = path_config
        if path_weights is None:
            path_weights = '/home/ubuntu/workspace/tabpfn_weights/tabforestpfn.pt'
        self.path_weights = path_weights

    def get_model_cls(self):
        if self.problem_type in ['binary', 'multiclass']:
            model_cls = TabForestPFNClassifier
        else:
            raise AssertionError(f"TabForestPFN does not support problem_type='{self.problem_type}'")
        return model_cls

    def _fit(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None, **kwargs):
        from tabularbench.config.config_run import ConfigRun
        import torch
        num_threads = None  # FIXME: Add param
        if num_threads is not None:
            torch.set_num_threads(num_threads)
        model_cls = self.get_model_cls()
        cfg = ConfigRun.load(Path(self.path_config))
        # cfg.output_dir = Path('results')
        if cfg.device is None:
            # cfg.device = 'cuda:0'
            cfg.device = 'cpu'

        cfg.hyperparams['max_epochs'] = self.max_epochs
        cfg.hyperparams['n_ensembles'] = self.n_ensembles

        if self.problem_type == "regression":
            cfg.task = "regression"
            n_classes = 0
        else:
            cfg.task = "classification"
            n_classes = len(y.unique())

        X = X.values.astype(np.float64)
        y = y.values
        if X_val is not None:
            X_val = X_val.values.astype(np.float64)
            y_val = y_val.values

        self.model = model_cls(
            cfg=cfg, n_classes=n_classes, split_val=self.split_val, path_to_weights=self.path_weights,
        )
        self.model.fit(
            X=X,
            y=y,
            X_val=X_val,
            y_val=y_val,
        )
        return self

    def _predict(self, X: pd.DataFrame) -> pd.Series:
        y_pred = self.model.predict(X.values.astype(np.float64))
        return y_pred

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        y_pred_proba = self.model.predict_proba(X.values.astype(np.float64))
        return y_pred_proba
