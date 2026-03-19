from __future__ import annotations

APP_NAME = "PD Data Analysis"

SUPPORTED_FORMATS = {
    "long_single": "Long / single timepoint",
    "long_time": "Long / time series",
    "wide_time": "Wide / time series",
    "replicate": "Replicate columns",
}

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

EFFECT_SIZE_DEFAULTS = {
    "one_way_anova": "omega_squared",
    "welch_anova": "omega_squared",
    "rm_anova": "partial_eta_squared",
    "mixed_anova": "partial_eta_squared",
    "pairwise_parametric": "hedges_g",
    "pairwise_nonparametric": "rank_biserial",
    "mixedlm": "beta_se_ci",
}
