from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class RepeatedStructureSummary:
    detected: bool = False
    has_subject_column: bool = False
    has_time_column: bool = False
    n_time_levels: int = 0
    n_subjects_with_multiple_timepoints: int = 0
    recommended_data_type: str = "cross"
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class BalanceSummary:
    is_balanced: bool = True
    has_missing_repeated_cells: bool = False
    subject_observation_counts: dict[str, int] = field(default_factory=dict)
    n_subjects_per_group: dict[str, int] = field(default_factory=dict)
    n_complete_subjects_per_group: dict[str, int] = field(default_factory=dict)
    missingness_info: dict = field(default_factory=dict)
    expected_time_levels: int = 0
    incomplete_subjects_by_group: dict[str, list[str]] = field(default_factory=dict)
    missing_repeated_cells: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class AssumptionSummary:
    normality: dict = field(default_factory=dict)
    levene: dict = field(default_factory=dict)
    sphericity: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class OutlierFlag:
    row_id: str
    subject_id: str | None = None
    group: str | None = None
    value: float | None = None
    method: str = ""
    statistic: float | None = None
    threshold: float | str | None = None
    pvalue: float | None = None
    flag_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class OutlierSummary:
    detected: bool = False
    method: str | None = None
    label: str | None = None
    flagged_count: int = 0
    sensitivity_run_available: bool = False
    handling_mode: str = "include_all"
    flags: list[OutlierFlag] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["flags"] = [flag.to_dict() if hasattr(flag, "to_dict") else flag for flag in self.flags]
        return payload


@dataclass(slots=True)
class CombinedDiagnosticsSummary:
    repeated_structure: RepeatedStructureSummary = field(default_factory=RepeatedStructureSummary)
    balance: BalanceSummary = field(default_factory=BalanceSummary)
    assumptions: AssumptionSummary = field(default_factory=AssumptionSummary)
    outliers: OutlierSummary = field(default_factory=OutlierSummary)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)
