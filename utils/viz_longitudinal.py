from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def make_figure(
    df: pd.DataFrame,
    result: dict,
    config: dict,
) -> go.Figure:
    fig = go.Figure()
    dv_col = result.get("dv_col") or next((col for col in df.columns if col.startswith("value_")), None)
    if dv_col is None or dv_col not in df.columns or "time" not in df.columns:
        return fig

    summary = df.groupby(["group", "time"], dropna=False)[dv_col].mean().reset_index()
    for group_name, group_df in summary.groupby("group", dropna=False, sort=False):
        fig.add_trace(
            go.Scatter(
                x=group_df["time"],
                y=group_df[dv_col],
                mode="lines+markers",
                name=str(group_name),
            )
        )

    _add_longitudinal_annotations(fig, summary, dv_col, result.get("star_map", []))
    fig.update_layout(template=config.get("template", "plotly_white"), title=f"Longitudinal plot: {dv_col}")
    return fig


def make_multi_biomarker_figure(
    df: pd.DataFrame,
    results_by_dv: dict[str, dict],
    config: dict,
) -> go.Figure:
    return make_figure(df=df, result=next(iter(results_by_dv.values()), {}), config=config)


def _add_longitudinal_annotations(fig: go.Figure, summary: pd.DataFrame, dv_col: str, star_map: list[dict]) -> None:
    if not star_map or summary.empty:
        return
    time_order = [str(time) for time in summary["time"].dropna().astype(str).unique().tolist()]
    max_y_by_time = summary.groupby("time")[dv_col].max().to_dict()
    time_offsets: dict[str, float] = {}
    for item in star_map:
        time_value = item.get("time")
        if time_value is None:
            continue
        time_key = str(time_value)
        if time_key not in time_order:
            continue
        base_y = float(max_y_by_time.get(time_value, summary[dv_col].max()))
        offset = time_offsets.get(time_key, max(abs(base_y) * 0.08, 0.1))
        fig.add_annotation(
            x=time_key,
            y=base_y + offset,
            text=item.get("label", "*"),
            showarrow=False,
        )
        time_offsets[time_key] = offset + max(abs(base_y) * 0.08, 0.1)
