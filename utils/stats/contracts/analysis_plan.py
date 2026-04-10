from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class AnalysisPlan:
    data_type: str
    design_family: str
    omnibus_method: str
    posthoc_method: str | None
    multiplicity_method: str | None
    engine: str
    effect_size_policy: str | None
    control_group: str | None = None
    reference_group: str | None = None
    factor2_col: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)
