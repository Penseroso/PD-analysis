from __future__ import annotations

from utils.stats.registry.effect_sizes import EFFECT_SIZE_DEFAULTS

APP_NAME = "PD Data Analysis"

SUPPORTED_FORMATS = {
    "long_single": "Long / single timepoint",
    "long_time": "Long / time series",
    "wide_time": "Wide / time series",
    "replicate": "Replicate columns",
}

# UI compatibility surface only.
# Internal method metadata and engine/default resolution now live under
# utils.stats.registry and should be treated as the source of truth.
ANALYSIS_METHODS = {
    "cross": [
        "auto",
        "one_way_anova",
        "welch_anova",
        "kruskal",
    ],
    "longitudinal": [
        "auto",
        "rm_anova",
        "mixed_anova",
        "friedman",
        "mixedlm",
    ],
}

DEFAULT_FIGURE_CONFIG = {
    "template": "plotly_white",
    "show_points": True,
    "show_sem": True,
    "spaghetti": False,
}

FORMAT_CONFIDENCE_THRESHOLD = 0.7
NUMERIC_SUCCESS_THRESHOLD = 0.8
