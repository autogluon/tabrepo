"""Methods to serialize tabular data into text.

Dependencies:
- tabstar
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tabstar.preprocessing.binning import fit_numerical_bins, transform_numerical_bins
from tabstar.preprocessing.dates import fit_date_encoders, transform_date_features
from tabstar.preprocessing.feat_types import (
    detect_numerical_features,
    transform_feature_types,
)
from tabstar.preprocessing.sparse import densify_objects
from tabstar.preprocessing.texts import replace_column_names
from tabstar.preprocessing.verbalize import verbalize_textual_features

if TYPE_CHECKING:
    import pandas as pd
    from pandas import DataFrame
    from sklearn.preprocessing import QuantileTransformer, StandardScaler
    from skrub import DatetimeEncoder


class CustomTabSTARVerbalizer:
    """Verbalizer from TabSTAR customized for our use case."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.date_transformers: dict[str, DatetimeEncoder] = {}
        self.numerical_transformers: dict[str, StandardScaler] = {}
        self.semantic_transformers: dict[str, QuantileTransformer] = {}
        self.constant_columns: list[str] = []

    def fit(self, X) -> CustomTabSTARVerbalizer:
        x = X.copy()
        self.assert_no_duplicate_columns(x)
        x, _ = densify_objects(x=x, y=None)
        self.date_transformers = fit_date_encoders(x=x)
        self.vprint(
            f"📅 Detected {len(self.date_transformers)} date features: {sorted(self.date_transformers)}"
        )
        x = transform_date_features(x=x, date_transformers=self.date_transformers)
        x, y = replace_column_names(x=x, y=None)
        numerical_features = detect_numerical_features(x)
        self.vprint(
            f"🔢 Detected {len(numerical_features)} numerical features: {sorted(numerical_features)}"
        )
        text_features = [col for col in x.columns if col not in numerical_features]
        self.vprint(
            f"📝 Detected {len(text_features)} textual features: {sorted(text_features)}"
        )
        x = transform_feature_types(x=x, numerical_features=numerical_features)
        self.constant_columns = [col for col in x.columns if x[col].nunique() == 1]
        for col in numerical_features:
            if col in self.constant_columns:
                continue
            self.semantic_transformers[col] = fit_numerical_bins(s=x[col])

        return self

    def transform(self, x: DataFrame) -> pd.DataFrame:
        x = x.copy()
        self.assert_no_duplicate_columns(x)
        x, _ = densify_objects(x=x, y=None)
        x = transform_date_features(x=x, date_transformers=self.date_transformers)
        x, _ = replace_column_names(x=x, y=None)
        num_cols = sorted(self.semantic_transformers)
        x = transform_feature_types(x=x, numerical_features=set(num_cols))
        x = verbalize_textual_features(x=x)
        x = x.drop(columns=self.constant_columns, errors="ignore")
        text_cols = [col for col in x.columns if col not in num_cols]
        x_txt = x[text_cols + num_cols].copy()

        for col in num_cols:
            x_txt[col] = transform_numerical_bins(
                s=x[col], scaler=self.semantic_transformers[col]
            )

        return x_txt

    @staticmethod
    def assert_no_duplicate_columns(x: DataFrame):
        if len(set(x.columns)) != len(x.columns):
            raise ValueError("Duplicate column names found in DataFrame!")

    def vprint(self, s: str):
        if self.verbose:
            print(s)
