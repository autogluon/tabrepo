from __future__ import annotations

import logging
import time

import numpy as np

from autogluon.common.features.types import R_INT, R_FLOAT, R_CATEGORY

logger = logging.getLogger(__name__)

from tabrepo.benchmark.models.ag.knn_new.knn_preprocessing import KNNPreprocessor

from autogluon.tabular.models.knn.knn_model import KNNModel

class KNNNewModel(KNNModel):
    """
    KNearestNeighbors model (scikit-learn): https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    """
    ag_key = "KNN_new"
    ag_name = "KNeighbors_new"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not hasattr(self, 'random_seed'):
            self.random_seed = 42

    def _preprocess(self, X, 
                    is_train: bool = False,
                    **kwargs):

        X = super(KNNModel, self)._preprocess(X, **kwargs)
        if is_train:
            X = self.knn_preprocessor.fit_transform(X)
        else:
            X = self.knn_preprocessor.transform(X)

        X = X.fillna(0).to_numpy(dtype=np.float32)
        return X

    def _set_default_params(self):
        default_params = {
            "weights": "distance",
            'scaler': 'standard',
            'cat_threshold': 10,
            'n_neighbors': 20,
            'p': 2,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            valid_raw_types=[R_INT, R_FLOAT, R_CATEGORY],  
            ignored_type_group_special=[],
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params

    def _fit(self, X, y, num_cpus=-1, time_limit=None, sample_weight=None, **kwargs):
        time_start = time.time()
        
        params = self._get_model_params()
        if "n_jobs" not in params:
            params["n_jobs"] = num_cpus
        if sample_weight is not None:  # TODO: support
            logger.log(15, "sample_weight not yet supported for KNNModel, this model will ignore them in training.")

        cat_cols = self._feature_metadata.get_features(
                    invalid_raw_types=["int", "float"]
                )
        self.knn_preprocessor = KNNPreprocessor(cat_threshold=params['cat_threshold'], categorical_features=cat_cols, numeric_strategy=params['scaler'])
        params.pop('cat_threshold')
        params.pop('scaler')

        X = self.preprocess(X, is_train=True)

        num_rows_max = len(X)

        # Fix for small datasets
        if num_rows_max < params['n_neighbors']:
            params['n_neighbors'] = min(params['n_neighbors'], num_rows_max - 1)
        
        # FIXME: v0.1 Must store final num rows for refit_full or else will use everything! Worst case refit_full could train far longer than the original model.
        if time_limit is None or num_rows_max <= 10000:
            self.model = self._get_model_type()(**params).fit(X, y)
        else:
            self.model = self._fit_with_samples(X=X, y=y, model_params=params, time_limit=time_limit - (time.time() - time_start))
