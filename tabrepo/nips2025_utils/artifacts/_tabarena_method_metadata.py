from __future__ import annotations

from tabrepo.nips2025_utils.artifacts.method_metadata import MethodMetadata
from tabrepo.nips2025_utils.artifacts.method_metadata_collection import MethodMetadataCollection
from tabrepo.nips2025_utils.artifacts._tabarena_method_metadata_2025_06_12 import (
    tabarena_method_metadata_map_2025_06_12
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

# TODO: Use a custom class instead of a dict. Don't treat `method` as a unique key.
#  Add table visualization of all available method_metadata.
tabarena_method_metadata_map: dict[str, MethodMetadata] = dict()
tabarena_method_metadata_map.update(tabarena_method_metadata_map_2025_06_12)
for method_metadata in methods_2025_09_03:
    assert method_metadata.method not in tabarena_method_metadata_map
    tabarena_method_metadata_map[method_metadata.method] = method_metadata

for method_metadata in methods_misc:
    assert method_metadata.method not in tabarena_method_metadata_map
    tabarena_method_metadata_map[method_metadata.method] = method_metadata

tabarena_method_metadata_collection = MethodMetadataCollection(
    method_metadata_lst=[v for _, v in tabarena_method_metadata_map.items()]
)
