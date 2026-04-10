from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OutlierMethodMetadata:
    id: str
    label: str
    family: str
    min_n: int
    detects_multiple_outliers: bool
    requires_continuous_data: bool = True
    notes: str = ""


OUTLIER_METHOD_REGISTRY: dict[str, OutlierMethodMetadata] = {
    "grubbs": OutlierMethodMetadata(
        id="grubbs",
        label="Grubbs test",
        family="parametric",
        min_n=3,
        detects_multiple_outliers=False,
        notes="Best suited to approximately normal data and a single extreme outlier per group.",
    ),
    "modified_zscore": OutlierMethodMetadata(
        id="modified_zscore",
        label="Modified Z-score",
        family="robust",
        min_n=3,
        detects_multiple_outliers=True,
        notes="Uses the median absolute deviation and works well as a general review flagging method.",
    ),
    "iqr": OutlierMethodMetadata(
        id="iqr",
        label="IQR rule",
        family="robust",
        min_n=4,
        detects_multiple_outliers=True,
        notes="Uses Tukey fences and is broadly applicable for exploratory outlier review.",
    ),
}


def get_outlier_method_metadata(method_id: str | None) -> OutlierMethodMetadata | None:
    if method_id is None:
        return None
    return OUTLIER_METHOD_REGISTRY.get(method_id)
