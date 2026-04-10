from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from utils.stats.contracts.analysis_plan import AnalysisPlan
from utils.stats.contracts.diagnostics import CombinedDiagnosticsSummary


@dataclass(slots=True)
class AnalysisResult:
    analysis_status: str
    plan: AnalysisPlan
    diagnostics: CombinedDiagnosticsSummary
    omnibus_table: pd.DataFrame | None = None
    pairwise_table: pd.DataFrame | None = None
    model_table: pd.DataFrame | None = None
    warnings: list[str] = field(default_factory=list)
    blocking_reasons: list[str] = field(default_factory=list)
    suggested_actions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["omnibus_table"] = self.omnibus_table
        payload["pairwise_table"] = self.pairwise_table
        payload["model_table"] = self.model_table
        return payload
