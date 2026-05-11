"""
Test suite for the RMS metric.
Run with:  python test_rms.py   (from src/evaluation/)
       or: python -m pytest src/evaluation/test_rms.py
"""

import copy, random, sys, os

# Add project root to sys.path so that src.evaluation.rms is a proper package
# (rms/core.py uses relative imports like `from ..row_types import ...`)
_proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

from src.evaluation.rms import compute_rms

random.seed(42)


def banner(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def show(result):
    print(f"  Chart type  : {result['chart_type']}")
    print(f"  Orientation : {result['orientation']}")
    print(f"  Precision   : {result['precision']:.4f}")
    print(f"  Recall      : {result['recall']:.4f}")
    print(f"  F1          : {result['f1']:.4f}")


# ─────────────────────────────────────────────────────────────
# Fixtures
#
# All fixtures include x_axis / y_axis metadata so that axis
# ranges are always available for range-normalised distance.
# Categorical axes are represented by null min/max.
# ─────────────────────────────────────────────────────────────

CAT_X = {
    "chart_title": "Cell Viability Study",
    "x_axis": {"min": None, "max": None},   # categorical axis
    "y_axis": {"min": 40, "max": 100},
    "data_points": [
        {"series_name": "5f",     "x_value": "Dark",  "y_value": 89.5},
        {"series_name": "AP5",    "x_value": "Dark",  "y_value": 93.0},
        {"series_name": "AP5+NO", "x_value": "Dark",  "y_value": 95.0},
        {"series_name": "5f",     "x_value": "Light", "y_value": 57.6},
        {"series_name": "AP5",    "x_value": "Light", "y_value": 85.9},
        {"series_name": "AP5+NO", "x_value": "Light", "y_value": 44.0},
    ],
}

SCATTER = {
    "chart_title": "Height vs Weight",
    "x_axis": {"min": 1.5, "max": 2.0},
    "y_axis": {"min": 40, "max": 100},
    "data_points": [
        {"series_name": "GroupA", "x_value": 1.70, "y_value": 65.0},
        {"series_name": "GroupA", "x_value": 1.75, "y_value": 72.0},
        {"series_name": "GroupB", "x_value": 1.60, "y_value": 55.0},
        {"series_name": "GroupB", "x_value": 1.80, "y_value": 85.0},
    ],
}

ERRORBAR = {
    "chart_title": None,
    "x_axis": {"min": None, "max": None},   # categorical (time points)
    "y_axis": {"min": 35, "max": 38},
    "data_points": [
        {"series_name": "FAT", "x_value": 5,  "y_value": {"min": 35.837, "median": 35.901, "max": 35.961}},
        {"series_name": "FAT", "x_value": 12, "y_value": {"min": 35.800, "median": 35.827, "max": 35.855}},
        {"series_name": "FAT", "x_value": 19, "y_value": {"min": 35.777, "median": 35.800, "max": 35.821}},
        {"series_name": "CON", "x_value": 5,  "y_value": {"min": 36.100, "median": 36.250, "max": 36.400}},
        {"series_name": "CON", "x_value": 12, "y_value": {"min": 36.050, "median": 36.180, "max": 36.310}},
    ],
}

BOXPLOT = {
    "chart_title": None,
    "x_axis": {"min": None, "max": None},   # categorical
    "y_axis": {"min": 0, "max": 40},
    "data_points": [
        {"series_name": "IPF-LC", "x_value": "CEA", "y_value": {"min": 2.0,  "q1": 3.5,  "median": 6.53, "q3": 8.7,  "max": 15.95}},
        {"series_name": "IPF",    "x_value": "CEA", "y_value": {"min": 0.37, "q1": 2.38, "median": 3.5,  "q3": 5.58, "max": 9.75}},
        {"series_name": "IPF-LC", "x_value": "CA",  "y_value": {"min": 5.0,  "q1": 8.0,  "median": 12.0, "q3": 20.0, "max": 35.0}},
        {"series_name": "IPF",    "x_value": "CA",  "y_value": {"min": 1.0,  "q1": 3.0,  "median": 5.5,  "q3": 9.0,  "max": 18.0}},
    ],
}

BUBBLE = {
    "chart_title": None,
    "x_axis": {"min": 0, "max": 0.06},
    "y_axis": {"min": None, "max": None},   # categorical
    "z_axis": {"min": 0, "max": 25},
    "w_axis": {"min": 0, "max": 8},
    "data_points": [
        {"series_name": "Main", "x_value": 0.0216, "y_value": "Protein export",     "z_value": 10, "w_value": 2.0},
        {"series_name": "Main", "x_value": 0.0312, "y_value": "Cell cycle",         "z_value": 15, "w_value": 3.5},
        {"series_name": "Main", "x_value": 0.0450, "y_value": "DNA repair",         "z_value": 20, "w_value": 6.5},
        {"series_name": "Main", "x_value": 0.0180, "y_value": "Apoptosis",          "z_value": 8,  "w_value": 1.0},
        {"series_name": "Main", "x_value": 0.0275, "y_value": "Signal transduction","z_value": 12, "w_value": 4.2},
    ],
}

BUBBLE_PARTIAL_Z = {
    "chart_title": None,
    "x_axis": {"min": 0, "max": 0.06},
    "y_axis": {"min": None, "max": None},
    "z_axis": {"min": 0, "max": 25},
    "w_axis": {"min": 0, "max": 8},
    "data_points": [
        {"series_name": "Main", "x_value": 0.0216, "y_value": "Protein export",     "z_value": None, "w_value": 2.0},
        {"series_name": "Main", "x_value": 0.0312, "y_value": "Cell cycle",         "z_value": None, "w_value": 3.5},
        {"series_name": "Main", "x_value": 0.0450, "y_value": "DNA repair",         "z_value": None, "w_value": 6.5},
        {"series_name": "Main", "x_value": 0.0180, "y_value": "Apoptosis",          "z_value": None, "w_value": 1.0},
        {"series_name": "Main", "x_value": 0.0275, "y_value": "Signal transduction","z_value": None, "w_value": 4.2},
    ],
}

BUBBLE_NO_ZW = {
    "chart_title": None,
    "x_axis": {"min": 0, "max": 0.06},
    "y_axis": {"min": None, "max": None},
    "data_points": [
        {"series_name": "Main", "x_value": 0.0216, "y_value": "Protein export",     "z_value": None, "w_value": None},
        {"series_name": "Main", "x_value": 0.0312, "y_value": "Cell cycle",         "z_value": None, "w_value": None},
        {"series_name": "Main", "x_value": 0.0450, "y_value": "DNA repair",         "z_value": None, "w_value": None},
    ],
}


# ─────────────────────────────────────────────────────────────
# bar chart tests
# ─────────────────────────────────────────────────────────────

banner("Test 1 – bar: identical (F1 = 1.0)")
r = compute_rms(CAT_X, CAT_X, chart_type="bar")
show(r)
assert abs(r["f1"] - 1.0) < 1e-6
assert r["chart_type"] == "bar"
print("  ✓ PASS")

banner("Test 2 – bar: permutation invariant")
shuffled = copy.deepcopy(CAT_X)
shuffled["data_points"] = list(reversed(shuffled["data_points"]))
r = compute_rms(shuffled, CAT_X, chart_type="bar")
show(r)
assert abs(r["f1"] - 1.0) < 1e-6
print("  ✓ PASS")

banner("Test 3 – bar: trasposizione disponibile se richiesta (try_transpose=True)")
transposed = copy.deepcopy(CAT_X)
for dp in transposed["data_points"]:
    dp["series_name"], dp["x_value"] = dp["x_value"], dp["series_name"]
r = compute_rms(transposed, CAT_X, chart_type="bar", try_transpose=True)
show(r)
assert r["f1"] > 0.99
assert r["orientation"] == "transposed"
print("  ✓ PASS")

banner("Test 3b – bar: orientamento sbagliato penalizzato senza try_transpose")
r = compute_rms(transposed, CAT_X, chart_type="bar")
show(r)
assert r["orientation"] == "normal"
assert r["f1"] < 0.99
print("  ✓ PASS")

banner("Test 4 – bar: small numeric noise (F1 > 0.60)")
noisy = copy.deepcopy(CAT_X)
for dp in noisy["data_points"]:
    dp["y_value"] *= 1 + random.uniform(-0.02, 0.02)
r = compute_rms(noisy, CAT_X, chart_type="bar")
show(r)
assert r["f1"] > 0.60
print("  ✓ PASS")

banner("Test 5 – bar: extra predicted entry lowers precision")
extra = copy.deepcopy(CAT_X)
extra["data_points"].append({"series_name": "EXTRA", "x_value": "Dark", "y_value": 50.0})
r = compute_rms(extra, CAT_X, chart_type="bar")
show(r)
assert r["precision"] < r["recall"]
print("  ✓ PASS")

banner("Test 6 – bar: missing predicted entry lowers recall")
missing = copy.deepcopy(CAT_X)
missing["data_points"] = missing["data_points"][:-1]
r = compute_rms(missing, CAT_X, chart_type="bar")
show(r)
assert r["recall"] < r["precision"]
print("  ✓ PASS")


# ─────────────────────────────────────────────────────────────
# line chart tests (same categorical_x format as bar)
# ─────────────────────────────────────────────────────────────

banner("Test 7 – line: identical (F1 = 1.0)")
r = compute_rms(CAT_X, CAT_X, chart_type="line")
show(r)
assert abs(r["f1"] - 1.0) < 1e-6
assert r["chart_type"] == "line"
print("  ✓ PASS")

banner("Test 8 – line: permutation invariant")
shuffled_line = copy.deepcopy(CAT_X)
shuffled_line["data_points"] = list(reversed(shuffled_line["data_points"]))
r = compute_rms(shuffled_line, CAT_X, chart_type="line")
show(r)
assert abs(r["f1"] - 1.0) < 1e-6
print("  ✓ PASS")

banner("Test 9 – line: trasposizione disponibile se richiesta (try_transpose=True)")
transposed_line = copy.deepcopy(CAT_X)
for dp in transposed_line["data_points"]:
    dp["series_name"], dp["x_value"] = dp["x_value"], dp["series_name"]
r = compute_rms(transposed_line, CAT_X, chart_type="line", try_transpose=True)
show(r)
assert r["f1"] > 0.99
print("  ✓ PASS")

banner("Test 9b – line: orientamento sbagliato penalizzato senza try_transpose")
r = compute_rms(transposed_line, CAT_X, chart_type="line")
show(r)
assert r["orientation"] == "normal"
assert r["f1"] < 0.99
print("  ✓ PASS")

banner("Test 10 – line: wrong numeric values lower F1")
wrong_line = copy.deepcopy(CAT_X)
for dp in wrong_line["data_points"]:
    dp["y_value"] = dp["y_value"] * 10   # 10x off → range-normalised D >> theta → 1.0
r = compute_rms(wrong_line, CAT_X, chart_type="line")
show(r)
assert r["f1"] < 0.5
print("  ✓ PASS")


# ─────────────────────────────────────────────────────────────
# scatter tests
# ─────────────────────────────────────────────────────────────

banner("Test 11 – scatter: identical (F1 = 1.0)")
r = compute_rms(SCATTER, SCATTER, chart_type="scatter")
show(r)
assert abs(r["f1"] - 1.0) < 1e-6
assert r["chart_type"] == "scatter"
print("  ✓ PASS")

banner("Test 12 – scatter: chart_type propagated in result")
r = compute_rms(SCATTER, SCATTER, chart_type="scatter")
show(r)
assert r["chart_type"] == "scatter"
assert abs(r["f1"] - 1.0) < 1e-6
print("  ✓ PASS")

banner("Test 13 – scatter: small x/y noise (F1 > 0.50)")
noisy_sc = copy.deepcopy(SCATTER)
for dp in noisy_sc["data_points"]:
    dp["x_value"] = dp["x_value"] * (1 + random.uniform(-0.02, 0.02))
    dp["y_value"] = dp["y_value"] * (1 + random.uniform(-0.02, 0.02))
r = compute_rms(noisy_sc, SCATTER, chart_type="scatter")
show(r)
assert r["f1"] > 0.50
print("  ✓ PASS")

banner("Test 14 – scatter: large error on one axis tanks F1")
bad_x = copy.deepcopy(SCATTER)
for dp in bad_x["data_points"]:
    dp["x_value"] = dp["x_value"] * 100   # 100x off → D >> theta → 1.0
r = compute_rms(bad_x, SCATTER, chart_type="scatter")
show(r)
assert r["f1"] <= 0.6
print("  ✓ PASS")

banner("Test 15 – scatter: transposition skipped (orientation always 'normal')")
r = compute_rms(SCATTER, SCATTER, chart_type="scatter")
assert r["orientation"] == "normal"
print(f"  orientation = {r['orientation']}")
print("  ✓ PASS")


# ─────────────────────────────────────────────────────────────
# metadata
# ─────────────────────────────────────────────────────────────

banner("Test 16 – wrong chart_title lowers F1")
wrong_title = copy.deepcopy(CAT_X)
wrong_title["chart_title"] = "Completely Wrong XYZ"
r_ok  = compute_rms(CAT_X, CAT_X, chart_type="bar")
r_bad = compute_rms(wrong_title, CAT_X, chart_type="bar")
print(f"  F1 correct title : {r_ok['f1']:.4f}")
print(f"  F1 wrong title   : {r_bad['f1']:.4f}")
assert r_bad["f1"] < r_ok["f1"]
print("  ✓ PASS")

banner("Test 17 – None title skipped (slightly lower recall)")
no_title = copy.deepcopy(CAT_X)
no_title["chart_title"] = None
r = compute_rms(no_title, CAT_X, chart_type="bar")
show(r)
assert r["f1"] < 1.0
assert r["recall"] < r["precision"]
print("  ✓ PASS")

print("\n" + "="*60)
print("  Tests 1-17 completed successfully.")
print("="*60)


# ─────────────────────────────────────────────────────────────
# Errorbar tests
# ─────────────────────────────────────────────────────────────

banner("Test 18 – errorbar: identical (F1 = 1.0)")
r = compute_rms(ERRORBAR, ERRORBAR, chart_type="errorpoint")
show(r)
assert abs(r["f1"] - 1.0) < 1e-6
print("  ✓ PASS")

banner("Test 19 – errorbar: permutation invariant")
shuffled_eb = copy.deepcopy(ERRORBAR)
shuffled_eb["data_points"] = list(reversed(shuffled_eb["data_points"]))
r = compute_rms(shuffled_eb, ERRORBAR, chart_type="errorpoint")
show(r)
assert abs(r["f1"] - 1.0) < 1e-6
print("  ✓ PASS")

banner("Test 20 – errorbar: small noise on all fields (F1 > 0.20)")
noisy_eb = copy.deepcopy(ERRORBAR)
for dp in noisy_eb["data_points"]:
    for k in dp["y_value"]:
        dp["y_value"][k] *= 1 + random.uniform(-0.005, 0.005)
r = compute_rms(noisy_eb, ERRORBAR, chart_type="errorpoint")
show(r)
assert r["f1"] > 0.20
print("  ✓ PASS")

banner("Test 21 – errorbar: wrong median tanks F1")
wrong_eb = copy.deepcopy(ERRORBAR)
for dp in wrong_eb["data_points"]:
    dp["y_value"]["median"] *= 2   # 100% error → D >> theta → clamped to 1.0
r = compute_rms(wrong_eb, ERRORBAR, chart_type="errorpoint")
show(r)
assert r["f1"] < 0.8
print("  ✓ PASS")

banner("Test 22 – boxplot: identical (F1 = 1.0)")
r = compute_rms(BOXPLOT, BOXPLOT, chart_type="box")
show(r)
assert abs(r["f1"] - 1.0) < 1e-6
print("  ✓ PASS")

banner("Test 23 – boxplot: permutation invariant")
shuffled_bp = copy.deepcopy(BOXPLOT)
shuffled_bp["data_points"] = list(reversed(shuffled_bp["data_points"]))
r = compute_rms(shuffled_bp, BOXPLOT, chart_type="box")
show(r)
assert abs(r["f1"] - 1.0) < 1e-6
print("  ✓ PASS")

banner("Test 24 – boxplot: small noise on all 5 fields (F1 > 0.97)")
noisy_bp = copy.deepcopy(BOXPLOT)
for dp in noisy_bp["data_points"]:
    for k in dp["y_value"]:
        dp["y_value"][k] *= 1 + random.uniform(-0.005, 0.005)
r = compute_rms(noisy_bp, BOXPLOT, chart_type="box")
show(r)
assert r["f1"] > 0.97
print("  ✓ PASS")

banner("Test 25 – boxplot vs errorbar mismatch (partial credit, F1 in (0, 1))")
eb_from_bp = copy.deepcopy(BOXPLOT)
for dp in eb_from_bp["data_points"]:
    dp["y_value"] = {k: v for k, v in dp["y_value"].items() if k in ("min", "median", "max")}
r = compute_rms(eb_from_bp, BOXPLOT, chart_type="box")
show(r)
assert 0.0 < r["f1"] < 1.0
print(f"  partial credit F1 = {r['f1']:.4f}  ✓ PASS")

print("\n" + "="*60)
print("  Tests 18-25 completed successfully.")
print("="*60)


# ─────────────────────────────────────────────────────────────
# Bubble chart tests
# ─────────────────────────────────────────────────────────────

banner("Test 26 – bubble: chart_type propagated in result")
r = compute_rms(BUBBLE, BUBBLE, chart_type="bubble")
assert r["chart_type"] == "bubble"
print(f"  chart_type = {r['chart_type']}")
print("  ✓ PASS")

banner("Test 27 – bubble: identical (F1 = 1.0)")
r = compute_rms(BUBBLE, BUBBLE, chart_type="bubble")
show(r)
assert abs(r["f1"] - 1.0) < 1e-6
assert r["chart_type"] == "bubble"
print("  ✓ PASS")

banner("Test 28 – bubble: permutation invariant")
shuffled_bub = copy.deepcopy(BUBBLE)
shuffled_bub["data_points"] = list(reversed(shuffled_bub["data_points"]))
r = compute_rms(shuffled_bub, BUBBLE, chart_type="bubble")
show(r)
assert abs(r["f1"] - 1.0) < 1e-6
print("  ✓ PASS")

banner("Test 29 – bubble: small noise on x, z, w (F1 > 0.90)")
noisy_bub = copy.deepcopy(BUBBLE)
for dp in noisy_bub["data_points"]:
    dp["x_value"] = dp["x_value"] * (1 + random.uniform(-0.01, 0.01))
    if dp["z_value"] is not None:
        dp["z_value"] = dp["z_value"] * (1 + random.uniform(-0.01, 0.01))
    if dp["w_value"] is not None:
        dp["w_value"] = dp["w_value"] * (1 + random.uniform(-0.01, 0.01))
r = compute_rms(noisy_bub, BUBBLE, chart_type="bubble")
show(r)
assert r["f1"] > 0.90
print("  ✓ PASS")

banner("Test 30 – bubble: wrong x tanks F1 (large error)")
wrong_x = copy.deepcopy(BUBBLE)
for dp in wrong_x["data_points"]:
    dp["x_value"] = dp["x_value"] * 100
r = compute_rms(wrong_x, BUBBLE, chart_type="bubble")
show(r)
assert r["f1"] < 0.7
print("  ✓ PASS")

banner("Test 31 – bubble: z_value=null in predicted → penalised vs target with z")
r = compute_rms(BUBBLE_PARTIAL_Z, BUBBLE, chart_type="bubble")
show(r)
assert r["f1"] < 1.0
print(f"  F1 with missing z = {r['f1']:.4f}  ✓ PASS")

banner("Test 32 – bubble: both z and w null in both → only x compared (F1 = 1.0)")
r = compute_rms(BUBBLE_NO_ZW, BUBBLE_NO_ZW, chart_type="bubble")
show(r)
assert abs(r["f1"] - 1.0) < 1e-6
print("  ✓ PASS")

banner("Test 33 – bubble: z null on both sides → no penalty for z")
r = compute_rms(BUBBLE_PARTIAL_Z, BUBBLE_PARTIAL_Z, chart_type="bubble")
show(r)
assert abs(r["f1"] - 1.0) < 1e-6, f"F1={r['f1']}"
print("  ✓ PASS")

banner("Test 34 – bubble: trasposizione disponibile se richiesta (try_transpose=True)")
transposed_bub = copy.deepcopy(BUBBLE)
for dp in transposed_bub["data_points"]:
    dp["series_name"], dp["y_value"] = dp["y_value"], dp["series_name"]
r = compute_rms(transposed_bub, BUBBLE, chart_type="bubble", try_transpose=True)
show(r)
assert r["f1"] > 0.99
print("  ✓ PASS")

banner("Test 34b – bubble: orientamento sbagliato penalizzato senza try_transpose")
r = compute_rms(transposed_bub, BUBBLE, chart_type="bubble")
show(r)
assert r["orientation"] == "normal"
assert r["f1"] < 0.99
print("  ✓ PASS")

print("\n" + "="*60)
print("  Tests 26-34 completed successfully.")
print("="*60)


# ─────────────────────────────────────────────────────────────
# DePlot reference test
# ─────────────────────────────────────────────────────────────

banner("Test 35 – DePlot reference: one wrong value → F1 = 6/7 ≈ 0.8571")
a = {
    "chart_title": "Cell Viability Study",
    "x_axis": {"min": None, "max": None},
    "y_axis": {"min": 40, "max": 100},
    "data_points": [
        {"series_name": "5f",     "x_value": "Dark",  "y_value": 89.5},
        {"series_name": "AP5",    "x_value": "Dark",  "y_value": 93},
        {"series_name": "AP5+NO", "x_value": "Dark",  "y_value": 95},
        {"series_name": "5f",     "x_value": "Light", "y_value": 57.6},
        {"series_name": "AP5",    "x_value": "Light", "y_value": 85.9},
        {"series_name": "AP5+NO", "x_value": "Light", "y_value": 44},   # ← predicted
    ],
}
b = {
    "chart_title": "Cell Viability Study",
    "x_axis": {"min": None, "max": None},
    "y_axis": {"min": 40, "max": 100},
    "data_points": [
        {"series_name": "5f",     "x_value": "Dark",  "y_value": 89.5},
        {"series_name": "AP5",    "x_value": "Dark",  "y_value": 93},
        {"series_name": "AP5+NO", "x_value": "Dark",  "y_value": 95},
        {"series_name": "5f",     "x_value": "Light", "y_value": 57.6},
        {"series_name": "AP5",    "x_value": "Light", "y_value": 85.9},
        {"series_name": "AP5+NO", "x_value": "Light", "y_value": 100},  # ← ground truth
    ],
}
result = compute_rms(a, b, chart_type="bar")
show(result)
assert round(result["f1"], 4) == 0.8571, f"FAIL: F1={result['f1']:.4f}, expected 0.8571"
print("  ✓ PASS")

print("\n" + "="*60)
print("  All 38 tests completed successfully.")
print("="*60)


# ─────────────────────────────────────────────────────────────
# scatter_degenerate + MetaRow regression
# ─────────────────────────────────────────────────────────────

banner("Test 39 – scatter: single series + chart_title, pred series_name mismatch")
_sc_gt = {
    "chart_title": "Height vs Weight",
    "x_axis": {"min": 147.0, "max": 202.0},
    "y_axis": {"min": 35.0,  "max": 131.0},
    "data_points": [
        {"series_name": "Main", "x_value": 150.5, "y_value": 45.0},
        {"series_name": "Main", "x_value": 152.0, "y_value": 54.0},
    ],
}
_sc_pred = {
    **_sc_gt,
    "data_points": [
        # model used "Data" instead of "Main"; values close but not exact
        {"series_name": "Data", "x_value": 150.1, "y_value": 45.6},
        {"series_name": "Data", "x_value": 151.7, "y_value": 54.1},
    ],
}
r = compute_rms(_sc_pred, _sc_gt, chart_type="scatter")
show(r)
assert r["f1"] > 0.80, f"Expected F1>0.80, got {r['f1']:.4f}"
print("  ✓ PASS")

print("\n" + "="*60)
print("  All 39 tests completed successfully.")
print("="*60)
