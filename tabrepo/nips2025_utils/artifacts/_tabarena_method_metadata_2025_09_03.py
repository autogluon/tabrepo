from __future__ import annotations

from tabrepo.nips2025_utils.artifacts.method_metadata import MethodMetadata

_common_kwargs = dict(
    artifact_name="tabarena-2025-09-03",
    s3_bucket="tabarena",
    s3_prefix="cache",
    upload_as_public=True,
    has_raw=True,
    has_processed=True,
    has_results=True,
    name_suffix=None,
)

# New methods (tabarena-2025-09-03)
ag_140_metadata = MethodMetadata(
    method="AutoGluon_v140",
    method_type="baseline",
    compute="gpu",
    date="2025-09-03",
    **_common_kwargs,
)
mitra_metadata = MethodMetadata(
    method="Mitra_GPU",
    method_type="config",
    compute="gpu",
    date="2025-09-03",
    ag_key="MITRA",
    config_default="Mitra_GPU_c1_BAG_L1",
    can_hpo=False,
    is_bag=True,
    **_common_kwargs,
)
limix_metadata = MethodMetadata(
    method="LimiX_GPU",
    method_type="config",
    compute="gpu",
    date="2025-09-03",
    ag_key="LIMIX",
    config_default="LimiX_GPU_c1_BAG_L1",
    can_hpo=False,
    is_bag=False,
    **_common_kwargs,
)
realmlp_gpu_metadata = MethodMetadata(
    method="RealMLP_GPU",
    method_type="config",
    compute="gpu",
    date="2025-09-03",
    ag_key="TA-REALMLP",
    config_default="RealMLP_GPU_c1_BAG_L1",
    can_hpo=True,
    is_bag=True,
    **_common_kwargs,
)
ebm_metadata = MethodMetadata(
    method="ExplainableBM",
    method_type="config",
    compute="cpu",
    date="2025-09-03",
    ag_key="EBM",
    config_default="ExplainableBM_c1_BAG_L1",
    can_hpo=True,
    is_bag=True,
    **_common_kwargs,
)
xrfm_metadata = MethodMetadata(
    method="xRFM_GPU",
    method_type="config",
    compute="gpu",
    date="2025-09-03",
    ag_key="XRFM",
    config_default="xRFM_GPU_c1_BAG_L1",
    can_hpo=True,
    is_bag=True,
    **_common_kwargs,
)
tabflex_metadata = MethodMetadata(
    method="TabFlex_GPU",
    method_type="config",
    compute="gpu",
    date="2025-09-03",
    ag_key="TABFLEX",
    config_default="TabFlex_GPU_c1_BAG_L1",
    can_hpo=False,
    is_bag=False,
    **_common_kwargs,
)
betatabpfn_metadata = MethodMetadata(
    method="BetaTabPFN_GPU",
    method_type="config",
    compute="gpu",
    date="2025-09-03",
    ag_key="BETA",
    config_default="BetaTabPFN_GPU_c1_BAG_L1",
    can_hpo=False,
    is_bag=True,
    **_common_kwargs,
)
