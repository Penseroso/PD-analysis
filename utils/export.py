from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go


TABLE_KEYS = ("omnibus", "posthoc_table", "fixed_effects", "contrast_table")


def build_export_bundle(
    normalized_df: pd.DataFrame,
    analysis_results: dict[str, dict],
    figure_objects: dict[str, go.Figure],
) -> dict[str, bytes | str | None]:
    html_chunks = []
    png_bytes: bytes | None = None

    for fig in figure_objects.values():
        html_chunks.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))
        if png_bytes is None:
            png_bytes = figure_to_png_bytes(fig)

    return {
        "html": "\n".join(html_chunks).encode("utf-8") if html_chunks else None,
        "png": png_bytes,
        "csv": normalized_df.to_csv(index=False) if normalized_df is not None else None,
        "stats_csv": results_to_csv_text(analysis_results),
    }


def figure_to_html_bytes(fig: go.Figure) -> bytes:
    return fig.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8")


def figure_to_png_bytes(fig: go.Figure) -> bytes | None:
    try:
        return fig.to_image(format="png")
    except Exception:
        return None


def results_to_csv_text(results: dict[str, dict]) -> str:
    frames: list[pd.DataFrame] = []
    for dv_col, result in results.items():
        summary_row = pd.DataFrame(
            [
                {
                    "section": "summary",
                    "dv_col": dv_col,
                    "analysis_status": result.get("analysis_status"),
                    "used_method": result.get("used_method", result.get("used_formula")),
                    "warnings": json.dumps(result.get("warnings", []), ensure_ascii=True),
                }
            ]
        )
        frames.append(summary_row)
        for key in TABLE_KEYS:
            table = result.get(key)
            if isinstance(table, pd.DataFrame) and not table.empty:
                export_table = table.copy()
                export_table.insert(0, "dv_col", dv_col)
                export_table.insert(1, "section", key)
                frames.append(export_table)
    if not frames:
        return ""
    return pd.concat(frames, ignore_index=True, sort=False).to_csv(index=False)
