from __future__ import annotations


COMPARISON_MODES = {
    "control_based": {
        "id": "control_based",
        "label": "Control-based",
    },
    "all_pairs": {
        "id": "all_pairs",
        "label": "All pairs",
    },
    "reference_based": {
        "id": "reference_based",
        "label": "Reference contrasts",
    },
}


def get_comparison_mode(mode_id: str | None) -> dict | None:
    if mode_id is None:
        return None
    return COMPARISON_MODES.get(mode_id)
