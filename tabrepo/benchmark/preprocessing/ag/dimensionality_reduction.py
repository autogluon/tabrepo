"""Feature generator that reduce the dimensionality of text embeddings."""

from __future__ import annotations

from typing import TYPE_CHECKING

from autogluon.common.features.feature_metadata import FeatureMetadata
from autogluon.common.features.types import (
    S_TEXT_EMBEDDING,
    S_TEXT_EMBEDDING_DR,
    R_FLOAT,
)
from autogluon.features.generators.abstract import AbstractFeatureGenerator
from sklearn.decomposition import PCA

if TYPE_CHECKING:
    import pandas as pd


class TextEmbeddingDimensionalityReductionFeatureGenerator(AbstractFeatureGenerator):
    """Used as model-specific preprocessing to reduce the dimensionality of text embeddings."""

    def __init__(self, pca_n_components: int = 30, **kwargs):
        super().__init__(**kwargs)
        self._pca = None
        self._pca_n_components = pca_n_components

    def _fit_transform(self, X: pd.DataFrame, **kwargs) -> tuple[pd.DataFrame, dict]:
        self._pca_kwargs = {"random_state": 0, "n_components": self._pca_n_components, "copy": False}
        self._pca = PCA(**self._pca_kwargs).set_output(transform="pandas")

        X_out = self._transform(X, is_train=True)
        type_family_groups_special = {S_TEXT_EMBEDDING_DR: list(X_out.columns)}

        return X_out, type_family_groups_special

    def _transform(self, X: pd.DataFrame, *, is_train: bool = False) -> pd.DataFrame:
        X = self._pca.fit_transform(X) if is_train else self._pca.transform(X)
        X.columns = self.get_feature_names(X.shape[1])
        return X

    @staticmethod
    def get_feature_names(n_components: int) -> list[str]:
        return [f"__text_embedding__pca_{i}" for i in range(n_components)]

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return {
            "required_special_types": [S_TEXT_EMBEDDING],
        }

    @staticmethod
    def get_infer_features_in_args_to_drop() -> dict:
        # Recommend to drop these features after PCA:
        return {"invalid_special_types": [S_TEXT_EMBEDDING]}

    def estimate_output_feature_metadata(self, feature_metadata_in: FeatureMetadata) -> FeatureMetadata:
        columns = self.get_feature_names(self._pca_n_components)
        return FeatureMetadata(
            type_map_raw=dict.fromkeys(columns, R_FLOAT),
            type_group_map_special={S_TEXT_EMBEDDING_DR: columns},
        )
