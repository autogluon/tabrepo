"""Feature generator that try to embed semantic information of text features.

Dependencies:
    - sentence_transformers
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from autogluon.common.features.types import (
    S_TEXT_EMBEDDING,
)
from autogluon.features.generators.abstract import AbstractFeatureGenerator
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from tabrepo.benchmark.preprocessing.ag.table_serialization import (
    CustomTabSTARVerbalizer,
)


class SemanticTextFeatureGenerator(AbstractFeatureGenerator):
    """Implements a simple custom feature generator to obtain semantic
    embeddings across rows.
    """

    def _fit_transform(self, X: pd.DataFrame, **kwargs) -> tuple[pd.DataFrame, dict]:
        """See parameters of the parent class AbstractFeatureGenerator for more details
        on the parameters.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # TODO: add multiprocessing here?
        self.verbalizer = CustomTabSTARVerbalizer()
        self._encoder_model = SentenceTransformer("intfloat/e5-small-v2", device=device)

        X_out = self._transform(X, is_train=True)
        return X_out, {S_TEXT_EMBEDDING: list(X_out.columns)}

    # TODO: https://sbert.net/examples/sentence_transformer/applications/embedding-quantization/README.html?
    def _transform(self, X: pd.DataFrame, *, is_train: bool = False) -> pd.DataFrame:
        """See parameters of the parent class AbstractFeatureGenerator for more details
        on the parameters.
        """
        if is_train:
            self.verbalizer.fit(X)

        # TODO: remove `f"Predictive Feature: {col}\nFeature Value:` and
        #   multiply by column name embedding instead?
        x = self.verbalizer.transform(X)

        # Embed the semantic information of each column
        semantic_embedding = []
        feature_names = []
        for col in tqdm(
            x.columns,
            desc="Encoding Semantic Information",
            total=len(x.columns),
        ):
            feature_emb = self._encoder_model.encode(x[col])
            semantic_embedding.append(feature_emb)
            feature_names.extend(
                [
                    f"__semantic_embedding_{i}__{col}"
                    for i in range(feature_emb.shape[1])
                ]
            )
        semantic_embedding = np.hstack(semantic_embedding)

        return pd.DataFrame(
            semantic_embedding,
            columns=feature_names,
        )

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        """Define the original feature's dtypes that this feature generator works on.

        See autogluon.features.FeatureMetadata.get_features for all options how to filter input data.
        See autogluon.features.types for all available raw and special types.
        """
        return {
            # Allow all features
            "valid_raw_types": None,
        }

    def _more_tags(self):
        return {"feature_interactions": True}
