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
    if dv_col is None or dv_col not in df.columns:
        return fig

    dv_label = result.get("dv_label") or dv_col

    for group_name, group_df in df.groupby("group", sort=False):
        fig.add_trace(
            go.Box(
                y=group_df[dv_col],
                name=str(group_name),`r`n                fillcolor="rgba(0,0,0,0)",
                boxpoints="all" if config.get("show_points", True) else False,
                jitter=0.25,
                pointpos=0,
            )
        )

    _add_cross_annotations(fig, df, dv_col, result.get("star_map", []))
    fig.update_layout(template=config.get("template", "plotly_white"), title=f"Cross-sectional plot: {dv_label}")
    return fig



def make_multi_biomarker_figure(
    df: pd.DataFrame,
    results_by_dv: dict[str, dict],
    config: dict,
) -> go.Figure:
    return make_figure(df=df, result=next(iter(results_by_dv.values()), {}), config=config)



def _add_cross_annotations(fig: go.Figure, df: pd.DataFrame, dv_col: str, star_map: list[dict]) -> None:
    if not star_map:
        return
    group_order = [str(group) for group in df["group"].dropna().astype(str).unique().tolist()]
    max_y = pd.to_numeric(df[dv_col], errors="coerce").max()
    if pd.isna(max_y):
        return
    step = max(abs(float(max_y)) * 0.08, 0.1)
    current_y = float(max_y) + step
    for item in star_map:
        group_a = str(item.get("group_a")) if item.get("group_a") is not None else None
        group_b = str(item.get("group_b")) if item.get("group_b") is not None else None
        if group_a not in group_order or group_b not in group_order:
            continue
        fig.add_annotation(
            x=(group_order.index(group_a) + group_order.index(group_b)) / 2,
            y=current_y,
            text=item.get("label", "*"),
            showarrow=False,
        )
        current_y += step

