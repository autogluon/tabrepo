from __future__ import annotations

from tabrepo.nips2025_utils.artifacts.method_metadata import MethodMetadata


common_kwargs = dict(
    artifact_name="tabarena-2025-06-12",
    date="2025-06-12",
    method_type="config",
    has_raw=True,
    has_processed=True,
    has_results=True,
)

cpu_kwargs = dict(
    compute="cpu",
    name_suffix=None,
    **common_kwargs,
)

gpu_kwargs = dict(
    compute="gpu",
    name_suffix="_GPU",
    **common_kwargs,
)

# Methods in this list will upload s3 artifacts privately (useful for storing results for unreleased models)
# If a method is not in this list, it will be public when uploaded.
methods_upload_as_private = []


# If the method should not be tuned/tuned+enesmbled in the simulator, for example, due to having only 1 config
methods_no_hpo = [
    "TabICL_GPU",
    "TabDPT_GPU",
]

# If the method was fit with bagging (8-fold)
# If not present in this list, the model could instead have been refit on the full data, ex: TabPFNv2
methods_is_bag = [
    "CatBoost",
    "Dummy",
    "ExplainableBM",
    "ExtraTrees",
    # "KNeighbors",
    "LightGBM",
    "LinearModel",
    "ModernNCA",
    "NeuralNetFastAI",
    "NeuralNetTorch",
    "RandomForest",
    "RealMLP",
    "TabM",
    "XGBoost",
    "Mitra_GPU",
    "ModernNCA_GPU",
    "RealMLP_GPU",
    # "TabDPT_GPU",
    # "TabICL_GPU",
    "TabM_GPU",
    # "TabPFNv2_GPU",
]


methods_ag_key_map = {
    "CatBoost": "CAT",
    "Dummy": "DUMMY",
    "ExplainableBM": "EBM",
    "ExtraTrees": "XT",
    "KNeighbors": "KNN",
    "LightGBM": "GBM",
    "LinearModel": "LR",
    "ModernNCA": "MNCA",
    "NeuralNetFastAI": "FASTAI",
    "NeuralNetTorch": "NN_TORCH",
    "RandomForest": "RF",
    "RealMLP": "REALMLP",
    "TabM": "TABM",
    "XGBoost": "XGB",

    "Mitra_GPU": "MITRA",
    "ModernNCA_GPU": "MNCA",
    "RealMLP_GPU": "REALMLP",
    "TabDPT_GPU": "TABDPT",
    "TabICL_GPU": "TABICL",
    "TabM_GPU": "TABM",
    "TabPFNv2_GPU": "TABPFNV2",
}

methods_config_default_map = {
    "CatBoost": "CatBoost_c1_BAG_L1",
    "Dummy": "Dummy_c1_BAG_L1",
    "ExplainableBM": "ExplainableBM_c1_BAG_L1",
    "ExtraTrees": "ExtraTrees_c1_BAG_L1",
    "KNeighbors": "KNeighbors_c1_BAG_L1",
    "LightGBM": "LightGBM_c1_BAG_L1",
    "LinearModel": "LinearModel_c1_BAG_L1",
    "ModernNCA": "ModernNCA_c1_BAG_L1",
    "NeuralNetFastAI": "NeuralNetFastAI_c1_BAG_L1",
    "NeuralNetTorch": "NeuralNetTorch_c1_BAG_L1",
    "RandomForest": "RandomForest_c1_BAG_L1",
    "RealMLP": "RealMLP_c1_BAG_L1",
    "TabM": "TabM_c1_BAG_L1",
    "XGBoost": "XGBoost_c1_BAG_L1",

    "Mitra_GPU": "Mitra_GPU_c1_BAG_L1",
    "ModernNCA_GPU": "ModernNCA_GPU_c1_BAG_L1",
    "RealMLP_GPU": "RealMLP_GPU_c1_BAG_L1",
    "TabDPT_GPU": "TabDPT_GPU_c1_BAG_L1",
    "TabICL_GPU": "TabICL_GPU_c1_BAG_L1",
    "TabM_GPU": "TabM_GPU_c1_BAG_L1",
    "TabPFNv2_GPU": "TabPFNv2_GPU_c1_BAG_L1",
}


methods_compute_map = {
    "CatBoost": "cpu",
    "Dummy": "cpu",
    "ExplainableBM": "cpu",
    "ExtraTrees": "cpu",
    "KNeighbors": "cpu",
    "LightGBM": "cpu",
    "LinearModel": "cpu",
    "ModernNCA": "cpu",
    "NeuralNetFastAI": "cpu",
    "NeuralNetTorch": "cpu",
    "RandomForest": "cpu",
    "RealMLP": "cpu",
    "TabM": "cpu",
    "XGBoost": "cpu",

    "Mitra_GPU": "gpu",
    "ModernNCA_GPU": "gpu",
    "RealMLP_GPU": "gpu",
    "TabDPT_GPU": "gpu",
    "TabICL_GPU": "gpu",
    "TabM_GPU": "gpu",
    "TabPFNv2_GPU": "gpu",

}


methods = [
    "CatBoost",
    "Dummy",
    "ExplainableBM",
    "ExtraTrees",
    "KNeighbors",
    "LightGBM",
    "LinearModel",
    "ModernNCA",
    "NeuralNetFastAI",
    "NeuralNetTorch",
    "RandomForest",
    "RealMLP",
    "TabM",
    "XGBoost",

    "Mitra_GPU",
    "ModernNCA_GPU",
    "RealMLP_GPU",
    "TabDPT_GPU",
    "TabICL_GPU",
    "TabM_GPU",
    "TabPFNv2_GPU",
]


tabarena_method_metadata_map: dict[str, MethodMetadata] = {}

for method in methods:
    compute_type = methods_compute_map[method]
    ag_key = methods_ag_key_map[method]
    config_default = methods_config_default_map[method]
    is_bag = method in methods_is_bag
    upload_as_public = method not in methods_upload_as_private
    assert compute_type in ["cpu", "gpu"]
    if compute_type == "cpu":
        method_kwargs = cpu_kwargs
    else:
        method_kwargs = gpu_kwargs
    if method in methods_no_hpo:
        can_hpo = False
    else:
        can_hpo = True

    method_metadata = MethodMetadata(
        method=method,
        config_default=config_default,
        ag_key=ag_key,
        is_bag=is_bag,
        can_hpo=can_hpo,
        upload_as_public=upload_as_public,
        **method_kwargs,
    )
    tabarena_method_metadata_map[method_metadata.method] = method_metadata


ag_130_metadata = MethodMetadata(
    method="AutoGluon_v130",
    artifact_name="tabarena-2025-06-12",
    date="2025-06-12",
    method_type="baseline",
    config_default="AutoGluon_v130_bq_4h8c",
    compute="cpu",
    has_raw=True,
    has_processed=True,
    has_results=True,
    upload_as_public=True,
)

tabarena_method_metadata_map[ag_130_metadata.method] = ag_130_metadata

portfolio_metadata = MethodMetadata(
    method="Portfolio-N200-4h",
    artifact_name="tabarena-2025-06-12",
    date="2025-06-12",
    method_type="portfolio",
    has_raw=False,
    has_processed=False,
    has_results=True,
    upload_as_public=True,
)

tabarena_method_metadata_map[portfolio_metadata.method] = portfolio_metadata
