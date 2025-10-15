from __future__ import annotations

from tabrepo.nips2025_utils.artifacts.method_metadata import MethodMetadata
from tabrepo.nips2025_utils.artifacts.method_metadata_collection import MethodMetadataCollection
from tabrepo.nips2025_utils.artifacts._tabarena_method_metadata_2025_06_12 import (
    methods_2025_06_12,
    methods_main_paper,
    methods_gpu_ablation,
)

from tabrepo.nips2025_utils.artifacts._tabarena_method_metadata_2025_09_03 import (
    ag_140_metadata,
    ebm_metadata,
    limix_metadata,
    mitra_metadata,
    realmlp_gpu_metadata,
    xrfm_metadata,
    tabflex_metadata,
    betatabpfn_metadata,
)
from tabrepo.nips2025_utils.artifacts._tabarena_method_metadata_misc import (
    gbm_aio_0808_metadata
)

methods_2025_09_03: list[MethodMetadata] = [
    ag_140_metadata,
    ebm_metadata,
    limix_metadata,
    mitra_metadata,
    realmlp_gpu_metadata,
    xrfm_metadata,
    betatabpfn_metadata,
    tabflex_metadata,
]

methods_misc: list[MethodMetadata] = [
    gbm_aio_0808_metadata,
]

replaced_methods = [
    "ExplainableBM",
    "RealMLP_GPU",
]
methods_2025_06_12_keep = [m for m in methods_2025_06_12 if m.method not in replaced_methods]


# The latest results for each method
tabarena_method_metadata_collection = MethodMetadataCollection(
    method_metadata_lst=methods_2025_06_12_keep + methods_2025_09_03 + methods_misc,
)

# All historical results for each method
tabarena_method_metadata_complete_collection = MethodMetadataCollection(
    method_metadata_lst=methods_2025_06_12 + methods_2025_09_03 + methods_misc,
)

# All historical results for each method
tabarena_method_metadata_2025_06_12_collection = MethodMetadataCollection(
    method_metadata_lst=methods_2025_06_12,
)

tabarena_method_metadata_2025_06_12_collection_main = MethodMetadataCollection(
    method_metadata_lst=[m for m in tabarena_method_metadata_2025_06_12_collection.method_metadata_lst if m.method in methods_main_paper]
)

tabarena_method_metadata_2025_06_12_collection_gpu_ablation = MethodMetadataCollection(
    method_metadata_lst=[m for m in tabarena_method_metadata_2025_06_12_collection.method_metadata_lst if m.method in methods_gpu_ablation]
)
