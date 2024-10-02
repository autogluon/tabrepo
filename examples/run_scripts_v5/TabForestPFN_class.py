from __future__ import annotations

import numpy as np
import pandas as pd
from abstract_class import AbstractExecModel

from pathlib import Path
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels


class TabForestPFN_sklearn(BaseEstimator, ClassifierMixin):

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

    def fit(self, X, y):
        from tabularbench.core.dataset_split import make_stratified_dataset_split
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        X = X.values.astype(np.float64)
        y = y.values
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y

        if self.split_val:
            X_train, X_valid, y_train, y_valid = make_stratified_dataset_split(X, y)
        else:
            X_train, X_valid, y_train, y_valid = X, X, y, y
        self.trainer.train(X_train, y_train, X_valid, y_valid)

        return self

    def predict(self, X):
        X = X.values.astype(np.float64)
        logits = self.trainer.predict(self.X_, self.y_, X)
        return logits.argmax(axis=1)

    def predict_proba(self, X):
        X = X.values.astype(np.float64)
        logits = self.trainer.predict(self.X_, self.y_, X)
        return np.exp(logits) / np.exp(logits).sum(axis=1)[:, None]


class CustomTabForestPFN(AbstractExecModel):
    def __init__(self, n_ensembles=1, max_epochs=0, split_val=False, path_config=None, path_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.n_ensembles = n_ensembles
        self.max_epochs = max_epochs
        self.split_val = split_val
        if path_config is None:
            path_config = "/home/ubuntu/workspace/code/TabForestPFN/outputs_done/foundation_mix_600k_finetune/test_categorical_classification/44156/#0/config_run.yaml"
        self.path_config = path_config
        if path_weights is None:
            path_weights = '/home/ubuntu/workspace/tabpfn_weights/tabforestpfn.pt'
        self.path_weights = path_weights


    def get_model_cls(self):
        if self.problem_type in ['binary', 'multiclass']:
            model_cls = TabForestPFN_sklearn
        else:
            raise AssertionError(f"TabForestPFN does not support problem_type='{self.problem_type}'")
        return model_cls

    def _fit(self, X: pd.DataFrame, y: pd.Series):
        from tabularbench.config.config_run import ConfigRun
        model_cls = self.get_model_cls()
        cfg = ConfigRun.load(Path(self.path_config))
        cfg.output_dir = Path('results')
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

        self.model = model_cls(
            cfg=cfg, n_classes=n_classes, split_val=self.split_val, path_to_weights=self.path_weights,
        )
        self.model.fit(
            X=X,
            y=y,
        )
        return self

    def _predict(self, X: pd.DataFrame) -> pd.Series:
        y_pred = self.model.predict(X)
        return pd.Series(y_pred, index=X.index)

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        y_pred_proba = self.model.predict_proba(X)
        y_pred_proba = pd.DataFrame(y_pred_proba, columns=self.model.classes_, index=X.index)
        return y_pred_proba
