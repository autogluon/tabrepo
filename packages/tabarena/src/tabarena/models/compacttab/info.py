from __future__ import annotations

from tabarena.models._method_metadata import ModelDescriptor
from tabarena.models._model_info import ModelInfo
from tabarena.models.compacttab.hpo import gen_compacttab, gen_compacttab_medium
from tabarena.models.compacttab.model import CompactTabFoundationMediumModel, CompactTabFoundationModel

_PIP_EXTRA = (
    "structured-data-models @ git+https://github.com/NVIDIA/structured-data-models.git@98f61289c7a4e1bce3b33771223ac2e123b63f19",
)


def _descriptor(display_name: str) -> ModelDescriptor:
    return ModelDescriptor(
        display_name=display_name,
        compute="gpu",
        is_bag=False,
        reference_url="https://github.com/DakshBhalala/TBDML",
        license="Apache-2.0 (code); OpenMDW-1.1 (weights)",
        date_introduced="2026-10-20",
    )


compacttab_descriptor = _descriptor("CompactTabFoundation")
compacttab_medium_descriptor = _descriptor("CompactTabFoundation-Medium")

_tabarena_run_kwargs = dict(
    can_hpo=False,
    date="2026-10-20",
    verified=False,
    cache_type="r2",
    validation_protocol="8x1",
    suite="tabarena-2026-10-20",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
)

compacttab_method_metadata = compacttab_descriptor.method_metadata(
    method="CompactTabFoundation",
    ag_key="TA-COMPACTTAB",
    config_default="CompactTabFoundation_c1_default_BAG_L1",
    **_tabarena_run_kwargs,
)

compacttab_medium_method_metadata = compacttab_medium_descriptor.method_metadata(
    method="CompactTabFoundation-Medium",
    ag_key="TA-COMPACTTAB-MEDIUM",
    config_default="CompactTabFoundation-Medium_c1_default_BAG_L1",
    **_tabarena_run_kwargs,
)

compacttab_info = ModelInfo(
    model_cls=CompactTabFoundationModel,
    search_space=gen_compacttab,
    method_metadata=compacttab_method_metadata,
    pip_extra=_PIP_EXTRA,
)

compacttab_medium_info = ModelInfo(
    model_cls=CompactTabFoundationMediumModel,
    search_space=gen_compacttab_medium,
    method_metadata=compacttab_medium_method_metadata,
    pip_extra=_PIP_EXTRA,
)
