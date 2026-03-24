from __future__ import annotations

from io import BytesIO
import json
import zipfile

import pandas as pd
import plotly.graph_objects as go


TABLE_KEYS = ("omnibus", "posthoc_table", "fixed_effects", "contrast_table")



def build_export_bundle(
    normalized_df: pd.DataFrame,
    analysis_results: dict[str, dict],
    figure_objects: dict[str, go.Figure],
) -> dict[str, bytes | str | None | list[str]]:
    html_chunks = []
    png_assets: list[tuple[str, bytes]] = []
    warnings: list[str] = []

    for dv_col, fig in figure_objects.items():
        html_chunks.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))
        png_bytes = figure_to_png_bytes(fig)
        if png_bytes is None:
            warnings.append(f"PNG export failed for '{dv_col}'.")
            continue
        png_assets.append((f"{dv_col}.png", png_bytes))

    png_bytes: bytes | None = None
    png_name: str | None = None
    png_mime: str | None = None
    if len(png_assets) == 1:
        png_name, png_bytes = png_assets[0]
        png_mime = "image/png"
    elif len(png_assets) > 1:
        png_bytes = _build_png_zip(png_assets)
        png_name = "figures.zip"
        png_mime = "application/zip"

    return {
        "html": "\n".join(html_chunks).encode("utf-8") if html_chunks else None,
        "png": png_bytes,
        "png_name": png_name,
        "png_mime": png_mime,
        "csv": normalized_df.to_csv(index=False) if normalized_df is not None else None,
        "stats_csv": results_to_csv_text(analysis_results),
        "warnings": warnings,
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
                    "used_formula": result.get("used_formula"),
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



def _build_png_zip(png_assets: list[tuple[str, bytes]]) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_name, png_bytes in png_assets:
            archive.writestr(file_name, png_bytes)
    return buffer.getvalue()
