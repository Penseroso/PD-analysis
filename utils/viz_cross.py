from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def make_figure(
    df: pd.DataFrame,
    result: dict,
    config: dict,
) -> go.Figure:
    fig = go.Figure()
    value_cols = [col for col in df.columns if col.startswith("value_")]
    if not value_cols:
        return fig

    dv_col = value_cols[0]
    for group_name, group_df in df.groupby("group"):
        fig.add_trace(
            go.Box(
                y=group_df[dv_col],
                name=str(group_name),
                boxpoints="all" if config.get("show_points", True) else False,
                jitter=0.25,
                pointpos=0,
            )
        )
    fig.update_layout(template=config.get("template", "plotly_white"), title=f"Cross-sectional plot: {dv_col}")
    return fig


def make_multi_biomarker_figure(
    df: pd.DataFrame,
    results_by_dv: dict[str, dict],
    config: dict,
) -> go.Figure:
    return make_figure(df=df, result=next(iter(results_by_dv.values()), {}), config=config)
