from __future__ import annotations

from tabrepo.nips2025_utils.artifacts.method_metadata import MethodMetadata


# New methods (tabarena-2025-09-03)
ag_140_metadata = MethodMetadata(
    method="AutoGluon_v140",
    artifact_name="tabarena-2025-09-03",
    date="2025-09-03",
    method_type="baseline",
    compute="gpu",
    has_raw=True,
    has_processed=True,
    has_results=True,
    upload_as_public=True,
)
limix_metadata = MethodMetadata(
    method="LimiX_GPU",
    artifact_name="tabarena-2025-09-03",
    date="2025-09-03",
    method_type="config",
    compute="gpu",
    ag_key="LIMIX",
    config_default="LimiX_GPU_c1_BAG_L1",
    name_suffix=None,
    has_raw=True,
    has_processed=True,
    has_results=True,
    upload_as_public=True,
    can_hpo=False,
)
realmlp_gpu_metadata = MethodMetadata(
    method="RealMLP_GPU",
    artifact_name="tabarena-2025-09-03",
    date="2025-09-03",
    method_type="config",
    compute="gpu",
    ag_key="TA-REALMLP",
    config_default="RealMLP_GPU_c1_BAG_L1",
    name_suffix=None,
    has_raw=True,
    has_processed=True,
    has_results=True,
    upload_as_public=True,
    can_hpo=False,
)
ebm_metadata = MethodMetadata(
    method="ExplainableBM",
    artifact_name="tabarena-2025-09-03",
    date="2025-09-03",
    method_type="config",
    compute="cpu",
    ag_key="EBM",
    config_default="ExplainableBM_c1_BAG_L1",
    name_suffix=None,
    has_raw=True,
    has_processed=True,
    has_results=True,
    upload_as_public=True,
    can_hpo=False,
)
