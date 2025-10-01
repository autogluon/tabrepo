from __future__ import annotations

from copy import deepcopy


def get_default_encoding_pipeline():
    """Return the default pipeline for encoding non-numerical or categorical data.

    The default pipeline handles:
        - Text Features
        - Date Time Features

    Text features are used to generate semantic and statistical embeddings.
    """
    from autogluon.features.generators import (
        AsTypeFeatureGenerator,
        FillNaFeatureGenerator,
    )
    from autogluon.features.generators.auto_ml_pipeline import PipelinePosition
    from tabrepo.benchmark.preprocessing.ag.date_time_features import DateTimeFeatureGenerator
    from tabrepo.benchmark.preprocessing.ag.statistical_text_embedding import StatisticalTextFeatureGenerator
    from tabrepo.benchmark.preprocessing.ag.semantic_text_embedding  import SemanticTextFeatureGenerator

    return {
        "custom_feature_generators": {
            PipelinePosition.AFTER_NUMERIC_FEATURES: [
                SemanticTextFeatureGenerator()
            ],
            PipelinePosition.AFTER_DATETIME_FEATURES: [
                DateTimeFeatureGenerator(),
            ],
            PipelinePosition.AFTER_TEXT_SPECIAL_FEATURES: [
                StatisticalTextFeatureGenerator()
            ],
        },
        # Disable default text ngram features.
        "enable_text_ngram_features": False,
        # Disable default datetime features.
        "enable_datetime_features": False,
        # Use default pre-generators but disabled conversion of bool features.
        "pre_generators": [
            AsTypeFeatureGenerator(convert_bool=False),
            FillNaFeatureGenerator(),
        ],
        "pre_enforce_types": False,
    }

def get_dimensionality_reduction_pipeline():
    from tabrepo.benchmark.preprocessing.ag.dimensionality_reduction import TextEmbeddingDimensionalityReductionFeatureGenerator
    from autogluon.common.features.types import S_TEXT_EMBEDDING
    from autogluon.features.generators.identity import IdentityFeatureGenerator

    return [
        # Passthrough for all non-text-embedding features
        IdentityFeatureGenerator(
            infer_features_in_args={
                "valid_raw_types": None,
                "invalid_special_types": [S_TEXT_EMBEDDING],
            }
        ),
        # PCA for text-embedding features
        TextEmbeddingDimensionalityReductionFeatureGenerator(),
    ]

def default_model_agnostic_pca(experiment):
    default_pipeline = get_default_encoding_pipeline()
    dr_pipeline = get_dimensionality_reduction_pipeline()

    # Enable model-agnostic dimensionality reduction to pipeline
    default_pipeline["post_generators"] = [dr_pipeline]

    # Add pipeline to experiment
    new_experiment = deepcopy(experiment)
    new_experiment.method_kwargs["fit_kwargs"][
        "_feature_generator_kwargs"
    ] = default_pipeline

    return new_experiment

def default_model_specific_pca(experiment):
    default_pipeline = get_default_encoding_pipeline()
    dr_pipeline = get_dimensionality_reduction_pipeline()

    # Add default pipeline to experiment
    new_experiment = deepcopy(experiment)
    new_experiment.method_kwargs["fit_kwargs"][
        "_feature_generator_kwargs"
    ] = default_pipeline

    # new_experiment.method_kwargs["init_kwargs"]["verbosity"] = 4

    # Add model-specific dimensionality reduction to experiment
    hps = new_experiment.method_kwargs["model_hyperparameters"]
    # TODO: figure out if this generalizes to having experiments with multiple models?
    #   ... and figure out if experiments with multiple models even exist
    # set to a large number such that max_features filter does not trigger before PCA
    hps["ag.model_specific_feature_generator_kwargs"] = {
            "feature_generators": [dr_pipeline],
        }
    new_experiment.method_kwargs["model_hyperparameters"] = hps
    return new_experiment

DEFAULT_PIPELINE_WITH_MODEL_AGNOSTIC_PCA = "D-PRE-MA_DR"
DEFAULT_PIPELINE_WITH_MODEL_SPECIFIC_PCA = "D-PRE-MS_DR"

PREPROCESSING_METHODS = {
    DEFAULT_PIPELINE_WITH_MODEL_AGNOSTIC_PCA: default_model_agnostic_pca,
    DEFAULT_PIPELINE_WITH_MODEL_SPECIFIC_PCA: default_model_specific_pca,
}