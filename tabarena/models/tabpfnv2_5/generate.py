from __future__ import annotations

from functools import partial

import numpy as np
from hyperopt import hp
from hyperopt.pyll import stochastic

from tabarena.benchmark.models.ag.tabpfnv2_5.tabpfnv2_5_model import RealTabPFNv25Model
from tabarena.utils.config_utils import CustomAGConfigGenerator


def enumerate_preprocess_transforms():
    transforms = []

    names_list = [
        ["quantile_uni_coarse"],
        ["quantile_norm_coarse"],
        ["kdi_uni"],
        ["kdi_alpha_0.3"],
        ["kdi_alpha_3.0"],
        ["none"],
        ["safepower", "quantile_uni"],
        ["none", "quantile_uni_coarse"],
        ["squashing_scaler_default", "quantile_uni_coarse"],
        ["squashing_scaler_default"],
    ]

    for names in names_list:
        for categorical_name in [
            "numeric",
            "ordinal_very_common_categories_shuffled",
            "none",
        ]:
            for append_original in [True, False]:
                for global_transformer_name in [None, "svd", "svd_quarter_components"]:
                    transforms += [
                        [
                            {
                                # Use "name" parameter as expected by TabPFN PreprocessorConfig
                                "name": name,
                                "global_transformer_name": global_transformer_name,
                                "categorical_name": categorical_name,
                                "append_original": append_original,
                            }
                            for name in names
                        ],
                    ]
    return transforms


def get_search_space_new(model_cls: RealTabPFNv25Model) -> dict:
    """Generate the full hyperopt search space for TabPFN optimization.

    Returns:
        Hyperopt search space dictionary
    """
    search_space = {
        # Custom HPs
        "model_type": hp.choice(
            "model_type",
            ["single", "dt_pfn"],
        ),
        "n_estimators": hp.choice("n_estimators", [4]),
        "max_depth": hp.choice("max_depth", [2, 3, 4, 5]),  # For Decision Tree TabPFN
        # -- Model HPs
        "average_before_softmax": hp.choice("average_before_softmax", [True, False]),
        "softmax_temperature": hp.choice(
            "softmax_temperature",
            [
                0.75,
                0.8,
                0.9,
                0.95,
                1.0,
                1.05,
            ],
        ),
        # Inference config
        "inference_config/FINGERPRINT_FEATURE": hp.choice(
            "FINGERPRINT_FEATURE",
            [True, False],
        ),
        "inference_config/PREPROCESS_TRANSFORMS": hp.choice(
            "PREPROCESS_TRANSFORMS",
            enumerate_preprocess_transforms(),
        ),
        "inference_config/POLYNOMIAL_FEATURES": hp.choice(
            "POLYNOMIAL_FEATURES",
            ["no"],  # Only use "no" to avoid polynomial feature computation errors
        ),
        "inference_config/OUTLIER_REMOVAL_STD": hp.choice(
            "OUTLIER_REMOVAL_STD",
            [None, 7.0, 12.0],
        ),
        "inference_config/MIN_UNIQUE_FOR_NUMERICAL_FEATURES": hp.choice(
            "MIN_UNIQUE_FOR_NUMERICAL_FEATURES", [1, 5, 10, 30]
        ),
    }

    search_space["classification_model_path"] = hp.choice(
        "classification_model_path",
        [
            model_cls.default_classification_model,
            *model_cls.extra_checkpoints_for_tuning("classification"),
        ],
    )

    # Zip model paths to ensure configs are not generated that only differ in combination
    clf_models = model_cls.extra_checkpoints_for_tuning("classification")
    reg_models = model_cls.extra_checkpoints_for_tuning("regression")
    zip_model_paths = [
        [model_cls.default_classification_model, model_cls.default_regression_model],
    ]
    n_clf_models = len(clf_models)
    n_reg_models = len(reg_models)
    for i in range(max(n_clf_models, n_reg_models)):
        zip_model_paths.append(
            [clf_models[min(i, n_clf_models - 1)], reg_models[min(i, n_reg_models - 1)]]
        )

    search_space["zip_model_path"] = hp.choice(
        "zip_model_path",
        zip_model_paths,
    )

    search_space["inference_config/REGRESSION_Y_PREPROCESS_TRANSFORMS"] = hp.choice(
        "REGRESSION_Y_PREPROCESS_TRANSFORMS",
        [
            (None,),
            (None, "safepower"),
            ("safepower",),
        ],
    )
    return search_space


def prepare_tabpfnv2_config(raw_config: dict) -> dict:
    """Set refit folds to True and convert tuples to lists."""
    return {k: list(v) if isinstance(v, tuple) else v for k, v in raw_config.items()}


def search_space_func_new(
    num_random_configs: int = 200, seed=1234, model_cls: RealTabPFNv25Model = None
) -> list[dict]:
    if model_cls is None:
        raise ValueError("model_cls must be provided!")
    search_space = get_search_space_new(model_cls=model_cls)
    rng = np.random.default_rng(seed)
    stochastic.sample(search_space, rng=rng)
    return [
        prepare_tabpfnv2_config(dict(stochastic.sample(search_space, rng=rng)))
        for _ in range(num_random_configs)
    ]


gen_realtabpfnv25 = CustomAGConfigGenerator(
    model_cls=RealTabPFNv25Model,
    search_space_func=partial(search_space_func_new, model_cls=RealTabPFNv25Model),
    manual_configs=[{}],
)

if __name__ == "__main__":
    from tabarena.benchmark.experiment import YamlExperimentSerializer

    print(
        YamlExperimentSerializer.to_yaml_str(
            experiments=gen_realtabpfnv25.generate_all_bag_experiments(
                num_random_configs=200
            ),
        ),
    )
