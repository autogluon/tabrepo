"""Feature generator that reduce the dimensionality of text embeddings."""

from __future__ import annotations

from typing import TYPE_CHECKING

from autogluon.common.features.types import (
    S_TEXT_EMBEDDING,
    S_TEXT_EMBEDDING_DR,
)
from autogluon.features.generators.abstract import AbstractFeatureGenerator
from sklearn.decomposition import PCA

if TYPE_CHECKING:
    import pandas as pd


class TextEmbeddingDimensionalityReductionFeatureGenerator(AbstractFeatureGenerator):
    """Used as model-specific preprocessing to reduce the dimensionality of text embeddings."""

    def _fit_transform(self, X: pd.DataFrame, **kwargs) -> tuple[pd.DataFrame, dict]:
        self._pca_kwargs = {"random_state": 0, "n_components": 30, "copy": False}
        self._pca = PCA(**self._pca_kwargs).set_output(transform="pandas")

        X_out = self._transform(X, is_train=True)
        type_family_groups_special = {S_TEXT_EMBEDDING_DR: list(X_out.columns)}

        return X_out, type_family_groups_special

    def _transform(self, X: pd.DataFrame, *, is_train: bool = False) -> pd.DataFrame:
        X = self._pca.fit_transform(X) if is_train else self._pca.transform(X)
        X.columns = [f"__text_embedding__pca_{i}" for i in range(X.shape[1])]
        return X

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return {
            "required_special_types": [S_TEXT_EMBEDDING],
        }

    @staticmethod
    def get_infer_features_in_args_to_drop() -> dict:
        # Recommend to also drop these features after PCA:
        return {"invalid_special_types": [S_TEXT_EMBEDDING]}
