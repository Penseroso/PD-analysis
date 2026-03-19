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
    if not value_cols or "time" not in df.columns:
        return fig

    dv_col = value_cols[0]
    summary = df.groupby(["group", "time"], dropna=False)[dv_col].mean().reset_index()
    for group_name, group_df in summary.groupby("group", dropna=False):
        fig.add_trace(
            go.Scatter(
                x=group_df["time"],
                y=group_df[dv_col],
                mode="lines+markers",
                name=str(group_name),
            )
        )
    fig.update_layout(template=config.get("template", "plotly_white"), title=f"Longitudinal plot: {dv_col}")
    return fig


def make_multi_biomarker_figure(
    df: pd.DataFrame,
    results_by_dv: dict[str, dict],
    config: dict,
) -> go.Figure:
    return make_figure(df=df, result=next(iter(results_by_dv.values()), {}), config=config)
