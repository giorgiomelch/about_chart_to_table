"""
HTML report generator for the sota_chart_to_table benchmark.

Shows parsed rows (GT table) and per-row similarity (match table) for each image.
Uses compute_rms(debug=True) from the new chart_types / rms API.
"""

import base64
import copy
import json
import os
import sys
import warnings
from pathlib import Path

# Allow running directly as: python src/evaluation/generate_reports.py
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from src.evaluation.evaluate import (
    estrai_basi,
    get_available_models,
    load_prediction,
    normalizza_valori,
)
from src.evaluation.rms import compute_rms
from src.evaluation.chart_types import get_parser
from src.evaluation.row_types import (
    BubbleRow, BoxRow, ChartRow, ErrorRow, MetaRow, ScatterRow, StandardRow,
)
from src.config import PREDICTIONS_DIR as PREDICTIONS_ROOT
from src.config import GROUNDTRUTH_DIR as GROUNDTRUTH_ROOT
from src.config import IMAGES_DIR as IMAGES_ROOT
from src.config import REPORTS_DIR as REPORTS_ROOT

_MODEL_PALETTE = [
    "#00d4ff",  # cyan
    "#82e0aa",  # green
    "#f0b27a",  # orange
    "#c39bd3",  # purple
    "#f9e79f",  # yellow
    "#85c1e9",  # light blue
    "#f1948a",  # salmon
    "#73c6b6",  # teal
    "#abebc6",  # mint
    "#d7bde2",  # lavender
]


def _model_color(model_name: str, model_names: list[str]) -> str:
    idx = model_names.index(model_name) if model_name in model_names else 0
    return _MODEL_PALETTE[idx % len(_MODEL_PALETTE)]


# ---------------------------------------------------------------------------
# Row display helpers
# ---------------------------------------------------------------------------

def _row_display(row: ChartRow) -> tuple[str, str, str]:
    """Return (series, key, value_str) for any ChartRow subtype."""
    if isinstance(row, StandardRow):
        return row.series, row.label, f"{row.value:.4g}"
    if isinstance(row, ScatterRow):
        return row.series, f"{row.x:.4g}", f"{row.y:.4g}"
    if isinstance(row, BoxRow):
        parts = [f"{k}={v:.3g}" for k, v in
                 [("min", row.min), ("q1", row.q1), ("med", row.median),
                  ("q3", row.q3), ("max", row.max)] if v is not None]
        return row.series, row.label, " | ".join(parts)
    if isinstance(row, ErrorRow):
        parts = [f"{k}={v:.3g}" for k, v in
                 [("min", row.min), ("med", row.median), ("max", row.max)] if v is not None]
        return row.series, row.label, " | ".join(parts)
    if isinstance(row, BubbleRow):
        extra = "".join(f" {k}={v:.3g}" for k, v in [("z", row.z), ("w", row.w)] if v is not None)
        return row.series, row.label, f"{row.value:.4g}{extra}"
    if isinstance(row, MetaRow):
        return "__meta__", row.field, str(row.value)
    return "", "?", str(row)


# ---------------------------------------------------------------------------
# Table styles
# ---------------------------------------------------------------------------

_TH = (
    "background:#2a2a2a; color:#999; padding:3px 8px; "
    "border-bottom:1px solid #555; text-align:left; white-space:nowrap;"
)
_TD = "padding:2px 8px; border-bottom:1px solid #222; vertical-align:top; white-space:nowrap;"
_SCROLL_WRAP = "overflow-x:auto; max-width:100%;"


# ---------------------------------------------------------------------------
# Parsed-rows table  (used for GT block)
# ---------------------------------------------------------------------------

def _parsed_rows_table(rows: list[ChartRow]) -> str:
    """Build an HTML table from a list of parsed ChartRow objects."""
    if not rows:
        return "<div style='color:#888; font-size:10px;'>Nessun dato parsato</div>"

    meta_rows = [r for r in rows if isinstance(r, MetaRow)]
    data_rows = [r for r in rows if not isinstance(r, MetaRow)]

    html = ""
    for m in meta_rows:
        html += (
            f"<div style='color:#aaa; font-size:10px; font-style:italic; margin-bottom:4px;'>"
            f"{m.field}: {m.value}</div>"
        )

    if not data_rows:
        return html + "<div style='color:#888; font-size:10px;'>Nessun dato</div>"

    rows_html = ""
    for row in data_rows:
        series, key, val = _row_display(row)
        rows_html += (
            f"<tr>"
            f"<td style='{_TD} color:#7fb3d3;'>{series}</td>"
            f"<td style='{_TD}'>{key}</td>"
            f"<td style='{_TD} color:#98d8a0;'>{val}</td>"
            f"</tr>"
        )

    table = (
        f"<table style='border-collapse:collapse; font-size:10px;'>"
        f"<tr>"
        f"<th style='{_TH}'>series</th>"
        f"<th style='{_TH}'>key</th>"
        f"<th style='{_TH}'>value</th>"
        f"</tr>"
        f"{rows_html}"
        f"</table>"
    )
    return html + f"<div style='{_SCROLL_WRAP}'>{table}</div>"


# ---------------------------------------------------------------------------
# Match table  (GT ↔ pred with per-row similarity)
# ---------------------------------------------------------------------------

def _sim_color(sim: float) -> str:
    if sim >= 0.8:
        return "#82e0aa"
    if sim >= 0.5:
        return "#f9e79f"
    if sim >= 0.2:
        return "#f0b27a"
    return "#e74c3c"


def _match_html_table(
    pred_rows: list[ChartRow],
    gt_rows: list[ChartRow],
    pairs: list,       # [(pred_idx, gt_idx, sim), ...]
) -> str:
    if not pred_rows and not gt_rows:
        return "<div style='color:#888; font-size:10px;'>Nessun dato</div>"

    display_pred = pred_rows

    matched_pred = {pi for pi, _, _ in pairs}
    matched_gt   = {ti for _, ti, _ in pairs}
    unmatched_gt   = [gt_rows[j]       for j in range(len(gt_rows))   if j not in matched_gt]
    unmatched_pred = [display_pred[i]  for i in range(len(pred_rows)) if i not in matched_pred]

    rows_html = ""
    empty3 = f"<td colspan='3' style='{_TD} color:#444; text-align:center;'>—</td>"

    for pi, ti, sim in sorted(pairs, key=lambda x: x[1]):
        p = display_pred[pi]
        t = gt_rows[ti]
        ts, tk, tv = _row_display(t)
        ps, pk, pv = _row_display(p)
        c = _sim_color(sim)
        rows_html += (
            f"<tr>"
            f"<td style='{_TD} color:#7fb3d3;'>{ts}</td>"
            f"<td style='{_TD}'>{tk}</td>"
            f"<td style='{_TD} color:#98d8a0;'>{tv}</td>"
            f"<td style='{_TD} color:#7fb3d3;'>{ps}</td>"
            f"<td style='{_TD} color:{c};'>{pk}</td>"
            f"<td style='{_TD} color:{c};'>{pv}</td>"
            f"<td style='{_TD} color:{c}; font-weight:bold; text-align:right;'>{sim:.2f}</td>"
            f"</tr>"
        )

    for t in unmatched_gt:
        ts, tk, tv = _row_display(t)
        rows_html += (
            f"<tr>"
            f"<td style='{_TD} color:#7fb3d3;'>{ts}</td>"
            f"<td style='{_TD}'>{tk}</td>"
            f"<td style='{_TD} color:#98d8a0;'>{tv}</td>"
            f"{empty3}"
            f"<td style='{_TD} color:#e74c3c; font-weight:bold; text-align:right;'>FN</td>"
            f"</tr>"
        )

    for p in unmatched_pred:
        ps, pk, pv = _row_display(p)
        rows_html += (
            f"<tr>"
            f"{empty3}"
            f"<td style='{_TD} color:#7fb3d3;'>{ps}</td>"
            f"<td style='{_TD} color:#e74c3c;'>{pk}</td>"
            f"<td style='{_TD} color:#e74c3c;'>{pv}</td>"
            f"<td style='{_TD} color:#e74c3c; font-weight:bold; text-align:right;'>FP</td>"
            f"</tr>"
        )

    if not rows_html:
        return "<div style='color:#888; font-size:10px;'>Nessun abbinamento</div>"

    table = (
        f"<table style='border-collapse:collapse; font-size:10px;'>"
        f"<tr>"
        f"<th style='{_TH}' colspan='3'>GT</th>"
        f"<th style='{_TH}' colspan='3'>Predizione</th>"
        f"<th style='{_TH}'>Sim</th>"
        f"</tr>"
        f"<tr>"
        f"<th style='{_TH}'>series</th><th style='{_TH}'>key</th><th style='{_TH}'>val</th>"
        f"<th style='{_TH}'>series</th><th style='{_TH}'>key</th><th style='{_TH}'>val</th>"
        f"<th style='{_TH}'></th>"
        f"</tr>"
        f"{rows_html}"
        f"</table>"
    )
    return f"<div style='{_SCROLL_WRAP}'>{table}</div>"



# ---------------------------------------------------------------------------
# Metrics bar
# ---------------------------------------------------------------------------

def _metrics_bar(result: dict) -> str:
    f1  = result["f1"]
    pre = result["precision"]
    rec = result["recall"]
    c   = _sim_color(f1)
    ori = result.get("orientation", "normal")
    return (
        f"<div style='color:{c}; font-size:10px; background:#0a0a0a; "
        f"padding:3px 6px; margin-bottom:6px; border-radius:3px; white-space:nowrap;'>"
        f"F1&nbsp;{f1*100:.1f} &nbsp;|&nbsp; P&nbsp;{pre*100:.1f} &nbsp;|&nbsp; "
        f"R&nbsp;{rec*100:.1f} &nbsp;|&nbsp; "
        f"<span style='color:#777'>{result['chart_type']} / {ori}</span>"
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Warnings display
# ---------------------------------------------------------------------------

def _warnings_html(warn_msgs: list[str]) -> str:
    if not warn_msgs:
        return ""
    items = "".join(
        f"<li style='margin:2px 0; color:#f0b27a;'>{w}</li>"
        for w in warn_msgs
    )
    return (
        f"<details style='margin-top:6px; font-size:10px;'>"
        f"<summary style='color:#f0b27a; cursor:pointer;'>⚠ {len(warn_msgs)} warning(s)</summary>"
        f"<ul style='margin:4px 0 0 12px; padding:0; color:#f0b27a;'>{items}</ul>"
        f"</details>"
    )


# ---------------------------------------------------------------------------
# Block wrapper — 3-column grid
# ---------------------------------------------------------------------------

_BLOCK_FLEX = "flex: 0 0 calc(33.33% - 12px); min-width: 300px; box-sizing: border-box;"
_BTN = (
    "font-size:9px; padding:1px 6px; border:1px solid #555; background:#2a2a2a; "
    "color:#aaa; cursor:pointer; border-radius:3px; font-family:monospace; "
    "vertical-align:middle; margin-left:8px;"
)


def _block(title: str, body_html: str, accent: str = "#888", bg: str = "#1e1e1e",
           json_uri: str = "") -> str:
    btn = (
        f"<a href='{json_uri}' target='_blank' style='text-decoration:none;'>"
        f"<button style='{_BTN}'>json</button></a>"
    ) if json_uri else ""
    return (
        f"<div style='{_BLOCK_FLEX} border:1px solid #444; border-top:2px solid {accent}; "
        f"padding:12px; border-radius:6px; background:{bg}; color:#e0e0e0; "
        f"font-family:monospace; font-size:12px; overflow:hidden;'>"
        f"<div style='margin-bottom:8px; color:{accent}; font-size:10px; font-weight:bold; "
        f"border-bottom:1px solid #333; padding-bottom:4px; text-transform:uppercase; "
        f"display:flex; align-items:center;'>"
        f"<span style='overflow:hidden; text-overflow:ellipsis; white-space:nowrap;'>{title}</span>"
        f"{btn}"
        f"</div>"
        f"{body_html}"
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Per-image section
# ---------------------------------------------------------------------------

def genera_sezione_immagine(
    img_path: Path,
    dataset_type: str,
    chart_class: str,
    model_names: list[str],
) -> str:
    rel_path = img_path.relative_to(IMAGES_ROOT / dataset_type / chart_class)
    gt_json_path = GROUNDTRUTH_ROOT / dataset_type / chart_class / rel_path.with_suffix(".json")

    with open(img_path, "rb") as fh:
        img_b64 = base64.b64encode(fh.read()).decode("utf-8")
    ext = img_path.suffix.lstrip(".").lower()
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext

    gt_data_raw: dict | None = None
    gt_data_norm: dict | None = None
    basi_gt: dict = {}
    gt_parse_warns: list[str] = []

    if gt_json_path.exists():
        with open(gt_json_path, "r", encoding="utf-8") as fh:
            gt_data_raw = json.load(fh)
        basi_gt = estrai_basi(gt_data_raw)
        gt_data_norm = normalizza_valori(copy.deepcopy(gt_data_raw), basi_gt)

    blocks: list[str] = []

    # Image block
    blocks.append(
        f"<div style='{_BLOCK_FLEX} border:1px solid #444; border-top:2px solid #555; "
        f"padding:10px; border-radius:6px; background:#2a2a2a; overflow:hidden;'>"
        f"<div style='margin-bottom:6px; color:#777; font-size:10px; "
        f"white-space:nowrap; overflow:hidden; text-overflow:ellipsis;'>{img_path.name}</div>"
        f"<img src='data:image/{mime};base64,{img_b64}' "
        f"style='width:100%; height:auto; border-radius:4px; display:block;'>"
        f"</div>"
    )

    # GT block — parsed rows from raw data
    if gt_data_raw:
        parser = get_parser(chart_class)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            gt_rows_raw = parser.parse(gt_data_raw)
        gt_parse_warns = [str(w.message) for w in caught]
        gt_body = _parsed_rows_table(gt_rows_raw) + _warnings_html(gt_parse_warns)
    else:
        gt_body = "<div style='color:#888'>GT Assente</div>"

    blocks.append(_block("Ground Truth", gt_body, accent="#5dade2", bg="#1a2535"))

    # Model blocks
    for model in model_names:
        accent = _model_color(model, model_names)
        pred_path = (
            PREDICTIONS_ROOT / model / dataset_type / chart_class / rel_path.with_suffix(".json")
        )

        pred_data_norm = load_prediction(pred_path, basi_gt)

        # Link to the actual prediction file (fallback to .txt for DePlot)
        actual_pred_path = pred_path if pred_path.exists() else pred_path.with_suffix(".txt")
        json_uri = actual_pred_path.as_uri() if actual_pred_path.exists() else ""

        if pred_data_norm is None:
            blocks.append(_block(model,
                "<div style='color:#888; font-size:10px;'>Predizione assente</div>",
                accent=accent, json_uri=json_uri))
            continue

        if gt_data_norm is None:
            blocks.append(_block(model,
                "<div style='color:#888; font-size:10px;'>GT assente</div>",
                accent=accent, json_uri=json_uri))
            continue

        pred_parse_warns: list[str] = []
        try:
            with warnings.catch_warnings(record=True) as pred_caught:
                warnings.simplefilter("always")
                try_transpose = model.lower() == "deplot"
                result = compute_rms(pred_data_norm, gt_data_norm,
                                     chart_type=chart_class, debug=True,
                                     try_transpose=try_transpose)
            pred_parse_warns = [str(w.message) for w in pred_caught]

            pairs = result[result["orientation"]]["pairs"]

            body = (
                _metrics_bar(result)
                + _match_html_table(result["pred_rows"], result["gt_rows"], pairs)
                + _warnings_html(pred_parse_warns)
            )
        except Exception as exc:
            body = (
                f"<div style='color:#e74c3c; font-size:10px;'>Errore metrica: {exc}</div>"
                + _warnings_html(pred_parse_warns)
            )

        blocks.append(_block(model, body, accent=accent, json_uri=json_uri))

    return (
        f"<div style='margin-bottom:50px; padding-bottom:20px; border-bottom:2px dashed #333;'>"
        f"<h3 style='color:#bbb; font-size:13px; margin-bottom:12px; font-family:monospace;'>"
        f"{img_path.name}</h3>"
        f"<div style='display:flex; flex-wrap:wrap; gap:16px;'>"
        f"{''.join(blocks)}"
        f"</div></div>"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_reports(chart_classes: list[str] | None = None) -> None:
    """Generate HTML reports. Pass chart_classes=['pie'] to restrict output."""
    model_names = get_available_models()

    legend_items = "".join(
        f"<span style='color:{_model_color(m, model_names)}; margin-right:16px;'>"
        f"&#9632; {m}</span>"
        for m in model_names
    )

    for dataset_type in ["arXiv", "PMCharts", "synthetic"]:
        base_img_dir = IMAGES_ROOT / dataset_type
        if not base_img_dir.exists():
            continue

        for chart_class_dir in sorted(base_img_dir.iterdir()):
            if not chart_class_dir.is_dir():
                continue

            chart_class = chart_class_dir.name
            if chart_classes and chart_class not in chart_classes:
                continue

            images = sorted(
                (f for f in chart_class_dir.rglob("*")
                 if f.suffix.lower() in (".jpg", ".png", ".jpeg")),
                key=lambda x: x.name.lower(),
            )
            if not images:
                continue

            html = (
                "<!DOCTYPE html><html><head><meta charset='utf-8'>"
                "<style>"
                "*, *::before, *::after { box-sizing: border-box; }"
                "body { background:#111; color:#ddd; padding:28px; "
                "       font-family:sans-serif; margin:0; }"
                "h2 { color:#00d4ff; border-bottom:2px solid #00d4ff; padding-bottom:8px; }"
                "</style></head><body>"
                f"<h2>Report Benchmark: {dataset_type.upper()} – {chart_class}</h2>"
                f"<div style='margin-bottom:24px; font-size:11px; font-family:monospace;'>"
                f"{legend_items}"
                f"<span style='color:#555; margin-left:12px;'>RMS tau=0.5 &theta;=0.1</span>"
                f"</div>"
            )

            for img_path in images:
                html += genera_sezione_immagine(
                    img_path, dataset_type, chart_class, model_names
                )

            html += "</body></html>"

            output_file = REPORTS_ROOT / dataset_type / f"report_{chart_class}.html"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(html, encoding="utf-8")
            print(f"Creato: {output_file}")


if __name__ == "__main__":
    generate_reports()
