"""Feature generators that try to embed the statistical signals of text features.

Dependencies:
    - skrub
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from autogluon.common.features.types import (
    S_IMAGE_BYTEARRAY,
    S_IMAGE_PATH,
    S_TEXT,
    S_TEXT_EMBEDDING,
)
from autogluon.features.generators.abstract import AbstractFeatureGenerator

if TYPE_CHECKING:
    import pandas as pd


class StatisticalTextFeatureGenerator(AbstractFeatureGenerator):
    """Generate a statistical embedding of text features using skrub."""

    LARGE_NUMBER_TO_AVOID_SKRUB_PCA = 1000000000

    def _fit_transform(self, X: pd.DataFrame, **kwargs) -> tuple[pd.DataFrame, dict]:
        from skrub import StringEncoder, TableVectorizer

        self._vectorizer = TableVectorizer(
            cardinality_threshold=20,
            low_cardinality=StringEncoder(
                n_components=self.LARGE_NUMBER_TO_AVOID_SKRUB_PCA, random_state=0
            ),
            high_cardinality=StringEncoder(
                n_components=self.LARGE_NUMBER_TO_AVOID_SKRUB_PCA, random_state=0
            ),
            numeric="drop",
            datetime="drop",
        )

        X_out = self._transform(X, is_train=True)
        type_family_groups_special = {S_TEXT_EMBEDDING: list(X_out.columns)}
        return X_out, type_family_groups_special

    def _transform(self, X: pd.DataFrame, *, is_train: bool = False) -> pd.DataFrame:
        X = X.astype(str)
        if is_train:
            X = self._vectorizer.fit_transform(X)
        else:
            X = self._vectorizer.transform(X)

        X.columns = [f"__statistical_embedding__{col_name}" for col_name in X.columns]

        return X

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return {
            "required_special_types": [S_TEXT],
            "invalid_special_types": [S_IMAGE_PATH, S_IMAGE_BYTEARRAY],
        }
