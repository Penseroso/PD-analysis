from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


TIME_PAIR_TYPES = {"longitudinal_time_pair_within_group"}
TIME_POINT_TYPES = {"longitudinal_group_pair_at_time", "mixedlm_reference_contrast"}



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

    summary = summary.copy()
    summary["time"] = summary["time"].astype(str)
    summary["group"] = summary["group"].astype(str)
    global_max = float(summary[dv_col].max()) if not summary.empty else 0.0
    y_step = max(abs(global_max) * 0.08, 0.1)

    time_peak = summary.groupby("time", sort=False)[dv_col].max().to_dict()
    time_group_peak = {
        (str(row["time"]), str(row["group"])): float(row[dv_col]) for _, row in summary.iterrows()
    }
    point_offsets: dict[str, float] = {}
    pair_offsets: dict[tuple[str, str, str | None], float] = {}

    for item in star_map:
        annotation_type = item.get("annotation_type")
        if annotation_type in TIME_POINT_TYPES:
            time_value = item.get("time")
            if time_value is None:
                continue
            time_key = str(time_value)
            base_y = float(time_peak.get(time_key, global_max))
            next_offset = point_offsets.get(time_key, y_step)
            label = item.get("label", "*")
            if item.get("group_a") and item.get("group_b"):
                label = f"{item.get('group_a')} vs {item.get('group_b')} {label}"
            fig.add_annotation(
                x=time_key,
                y=base_y + next_offset,
                text=label,
                showarrow=False,
            )
            point_offsets[time_key] = next_offset + y_step
            continue

        if annotation_type in TIME_PAIR_TYPES:
            time_a = item.get("time_a")
            time_b = item.get("time_b")
            if time_a is None or time_b is None:
                continue
            time_a_key = str(time_a)
            time_b_key = str(time_b)
            group_name = item.get("group")
            pair_key = tuple(sorted((time_a_key, time_b_key))) + (str(group_name) if group_name is not None else None,)
            base_y = max(
                _time_pair_peak(time_group_peak, time_peak, time_a_key, time_b_key, group_name),
                global_max,
            )
            next_offset = pair_offsets.get(pair_key, y_step)
            y = base_y + next_offset
            fig.add_shape(type="line", x0=time_a_key, x1=time_b_key, y0=y, y1=y, line={"color": "black", "width": 1})
            fig.add_shape(type="line", x0=time_a_key, x1=time_a_key, y0=y - y_step * 0.15, y1=y, line={"color": "black", "width": 1})
            fig.add_shape(type="line", x0=time_b_key, x1=time_b_key, y0=y - y_step * 0.15, y1=y, line={"color": "black", "width": 1})
            label = item.get("label", "*")
            if group_name is not None:
                label = f"{group_name} {label}"
            fig.add_annotation(x=time_b_key, y=y + y_step * 0.1, text=label, showarrow=False)
            pair_offsets[pair_key] = next_offset + y_step



def _time_pair_peak(
    time_group_peak: dict[tuple[str, str], float],
    time_peak: dict[str, float],
    time_a: str,
    time_b: str,
    group_name: str | None,
) -> float:
    if group_name is not None:
        return max(
            float(time_group_peak.get((time_a, str(group_name)), time_peak.get(time_a, 0.0))),
            float(time_group_peak.get((time_b, str(group_name)), time_peak.get(time_b, 0.0))),
        )
    return max(float(time_peak.get(time_a, 0.0)), float(time_peak.get(time_b, 0.0)))
