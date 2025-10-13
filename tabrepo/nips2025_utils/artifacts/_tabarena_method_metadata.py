from __future__ import annotations

from tabrepo.nips2025_utils.artifacts.method_metadata import MethodMetadata
from tabrepo.nips2025_utils.artifacts.method_metadata_collection import MethodMetadataCollection
from tabrepo.nips2025_utils.artifacts._tabarena_method_metadata_2025_06_12 import methods_2025_06_12

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

tabarena_method_metadata_collection = MethodMetadataCollection(
    method_metadata_lst=methods_2025_06_12 + methods_2025_09_03 + methods_misc,
)
