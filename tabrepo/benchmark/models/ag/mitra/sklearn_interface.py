import numpy as np
import time
import torch as th
import pandas as pd

from pathlib import Path
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ._internal.data.dataset_split import make_stratified_dataset_split
from ._internal.config.config_run import ConfigRun
from ._internal.core.trainer_finetune import TrainerFinetune
from ._internal.models.tab2d import Tab2D
from ._internal.config.enums import ModelName

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.metrics import mean_squared_error, r2_score

# Constants
DEFAULT_MODEL_TYPE = "Tab2D"
DEFAULT_DEVICE = "cuda"
DEFAULT_EPOCH = 10000
DEFAULT_BUDGET = 300
DEFAULT_ENSEMBLE = 1
DEFAULT_DIM = 512
DEFAULT_LAYERS = 12
DEFAULT_HEADS = 4
DEFAULT_CLASSES = 10
DEFAULT_VALIDATION_SPLIT = 0.2

class MitraBase(BaseEstimator):
    """Base class for Mitra models with common functionality."""
    
    def __init__(self, model_type=DEFAULT_MODEL_TYPE, n_estimators=DEFAULT_ENSEMBLE, 
                 device=DEFAULT_DEVICE, epoch=DEFAULT_EPOCH, budget=DEFAULT_BUDGET, state_dict=None):
        """
        Initialize the base Mitra model.
        
        Parameters
        ----------
        model_type : str, default="Tab2D"
            The type of model to use. Options: "Tab2D", "Tab2D_COL_ROW"
        n_estimators : int, default=1
            Number of models in the ensemble
        device : str, default="cuda"
            Device to run the model on
        epoch : int, default=0
            Number of epochs to train for
        state_dict : str, optional
            Path to the pretrained weights
        """
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.device = device
        self.epoch = epoch
        self.budget = budget
        self.state_dict = state_dict
        self.trainers = []
        self.models = []
        self.train_time = 0


    def _create_config(self, task, dim_output):
        cfg = ConfigRun(
            device=self.device,
            model_name=ModelName.TAB2D,
            seed=0,
            hyperparams={
                'dim_embedding': None,
                'early_stopping_data_split': 'VALID',
                'early_stopping_max_samples': 2048,
                'early_stopping_patience': 40,
                'grad_scaler_enabled': False,
                'grad_scaler_growth_interval': 1000,
                'grad_scaler_scale_init': 65536.0,
                'grad_scaler_scale_min': 65536.0,
                'label_smoothing': 0.0,
                'lr_scheduler': False,
                'lr_scheduler_patience': 25,
                'max_epochs': self.epoch,
                'max_samples_query': 1024,
                'max_samples_support': 8192,
                'optimizer': 'adamw',
                'lr': 0.0001,
                'weight_decay': 0.1,
                'warmup_steps': 1000,
                'path_to_weights': self.state_dict,
                'precision': 'bfloat16',
                'random_mirror_regression': True,
                'random_mirror_x': True,
                'shuffle_classes': self.n_estimators > 1,
                'shuffle_features': self.n_estimators > 1,
                'use_feature_count_scaling': False,
                'use_pretrained_weights': False,
                'use_quantile_transformer': False,
                'budget': self.budget,
            },
        )

        cfg.task = task
        cfg.hyperparams.update({
            'n_ensembles': self.n_estimators,
            'dim': DEFAULT_DIM,
            'dim_output': dim_output,
            'n_layers': DEFAULT_LAYERS,
            'n_heads': DEFAULT_HEADS,
            'regression_loss': 'mse',
        })

        return cfg, Tab2D

    
    def _split_data(self, X, y):
        """Split data into training and validation sets."""
        if hasattr(self, 'task') and self.task == 'classification':
            return make_stratified_dataset_split(X, y)
        else:
            # For regression, use random split
            val_indices = np.random.choice(range(len(X)), int(DEFAULT_VALIDATION_SPLIT * len(X)), replace=False).tolist()
            train_indices = [i for i in range(len(X)) if i not in val_indices]
            return X[train_indices], X[val_indices], y[train_indices], y[val_indices]
    
    def _train_ensemble(self, X_train, y_train, X_valid, y_valid, task, dim_output, n_classes=0):
        """Train the ensemble of models."""
        cfg, Tab2D = self._create_config(task, dim_output)
        self.trainers.clear()
        self.models.clear()
        
        self.train_time = 0
        for _ in range(self.n_estimators):
            model = Tab2D(
                dim=cfg.hyperparams['dim'],
                dim_output=dim_output,
                n_layers=cfg.hyperparams['n_layers'],
                n_heads=cfg.hyperparams['n_heads'],
                task=task.upper(),
                use_pretrained_weights=True,
                path_to_weights=Path(self.state_dict),
            )
            trainer = TrainerFinetune(cfg, model, n_classes=n_classes, device=self.device)

            start_time = time.time()
            trainer.train(X_train, y_train, X_valid, y_valid)
            end_time = time.time()

            self.trainers.append(trainer)
            self.models.append(model)
            self.train_time += end_time - start_time

        return self


class MitraClassifier(MitraBase, ClassifierMixin):
    """Classifier implementation of Mitra model."""

    def __init__(self, model_type=DEFAULT_MODEL_TYPE, n_estimators=DEFAULT_ENSEMBLE, 
                 device=DEFAULT_DEVICE, epoch=DEFAULT_EPOCH, state_dict='/fsx/xiyuanz/mix5_multi_cat.pt'):
        """Initialize the classifier."""
        super().__init__(model_type, n_estimators, device, epoch, state_dict)
        self.task = 'classification'
    
    def fit(self, X, y, X_val = None, y_val = None):
        """
        Fit the ensemble of models.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : object
            Returns self
        """

        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        self.X, self.y = X, y

        if X_val is not None and y_val is not None:
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            if isinstance(y_val, pd.Series):
                y_val = y_val.values
            X_train, X_valid, y_train, y_valid = X, X_val, y, y_val
        else:
            X_train, X_valid, y_train, y_valid = self._split_data(X, y)

        return self._train_ensemble(X_train, y_train, X_valid, y_valid, self.task, DEFAULT_CLASSES, n_classes=DEFAULT_CLASSES)

    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples
            
        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted classes
        """

        if isinstance(X, pd.DataFrame):
            X = X.values

        return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples
            
        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        preds = []
        for trainer in self.trainers:
            logits = trainer.predict(self.X, self.y, X)[...,:len(np.unique(self.y))] # Remove extra classes
            preds.append(np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)) # Softmax
        preds = sum(preds) / len(preds)  # Averaging ensemble predictions
        return preds


class MitraRegressor(MitraBase, RegressorMixin):
    """Regressor implementation of Mitra model."""

    def __init__(self, model_type=DEFAULT_MODEL_TYPE, n_estimators=DEFAULT_ENSEMBLE, 
                 device=DEFAULT_DEVICE, epoch=DEFAULT_EPOCH, state_dict='/fsx/xiyuanz/atticmix4reg.pt'):
        """Initialize the regressor."""
        super().__init__(model_type, n_estimators, device, epoch, state_dict)
        self.task = 'regression'

    def fit(self, X, y, X_val = None, y_val = None):
        """
        Fit the ensemble of models.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : object
            Returns self
        """

        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        self.X, self.y = X, y

        if X_val is not None and y_val is not None:
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            if isinstance(y_val, pd.Series):
                y_val = y_val.values
            X_train, X_valid, y_train, y_valid = X, X_val, y, y_val
        else:
            X_train, X_valid, y_train, y_valid = self._split_data(X, y)

        return self._train_ensemble(X_train, y_train, X_valid, y_valid, self.task, 1)

    def predict(self, X):
        """
        Predict regression target for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples
            
        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted values
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        preds = [trainer.predict(self.X, self.y, X) for trainer in self.trainers]
        return sum(preds) / len(preds)  # Averaging ensemble predictions


if __name__ == '__main__':

    # Load data
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    # Initialize a classifier
    clf = MitraClassifier(
        model_type=DEFAULT_MODEL_TYPE, 
        n_estimators=DEFAULT_ENSEMBLE, 
        device=DEFAULT_DEVICE, 
        epoch=1000, 
        state_dict='/home/yuyawang/checkpoints/mix5_multi_cat.pt'
    )
    clf.fit(X_train, y_train)

    # Predict probabilities
    prediction_probabilities = clf.predict_proba(X_test)
    print("ROC AUC:", roc_auc_score(y_test, prediction_probabilities[:, 1]))

    # Predict labels
    predictions = clf.predict(X_test)
    print("Accuracy", accuracy_score(y_test, predictions))

    # Load Boston Housing data
    df = fetch_openml(data_id=531, as_frame=True)  # Boston Housing dataset
    X = df.data
    y = df.target.astype(float)  # Ensure target is float for regression

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    X_train, X_test, y_train, y_test = X_train.values.astype(np.float64), X_test.values.astype(np.float64), y_train.values, y_test.values 

    # Initialize the regressor
    regressor = MitraRegressor(
        model_type=DEFAULT_MODEL_TYPE,
        n_estimators=DEFAULT_ENSEMBLE,
        device=DEFAULT_DEVICE,
        epoch=0,
        state_dict='/home/yuyawang/checkpoints/atticmix4reg.pt'
    ) 
    regressor.fit(X_train, y_train)

    # Predict on the test set
    predictions = regressor.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("Mean Squared Error (MSE):", mse)
    print("R² Score:", r2)