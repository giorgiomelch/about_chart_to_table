# annotate_charts

This folder contains the tools used to create the **ground-truth annotations** for the [SciChartBench](../SciChartBench/README.md) benchmark.

---

## Main tool: WebPlotDigitizer

Most annotations (bar, box, bubble, errorpoint, histogram, line, radar, scatter) were created with **WebPlotDigitizer**, a free web-based tool that requires no installation.

- URL: https://automeris.io/wpd/
- Workflow: load a chart image, calibrate the axes, click on data points, export as CSV or JSON.

---

## Specialised tool: `heatmap_labeler.py`

Heatmaps require a different approach: numeric values are extracted from cell colours by interpolating the **colorbar**. `heatmap_labeler.py` is a desktop application (Tkinter) that automates this process.

### Requirements

```bash
pip install pillow numpy
```

### Usage

```bash
python heatmap_labeler.py [image_path]
```

If no path is provided, a file-picker dialog opens.

### Workflow

1. **Select the colorbar**: drag a rectangle over the heatmap's colour bar.
2. **Enter Min and Max values**: the values corresponding to the two extremes of the colorbar.
3. **Click on cells**: the app interpolates the clicked colour and displays the corresponding numeric value.
4. **Export**: copy data as TSV or save the JSON file in the format required by the benchmark.

### Output JSON format

```json
{
  "chart_title": "...",
  "x_axis_label": "...",
  "y_axis_label": "...",
  "data_points": [
    {"series_name": "row_label", "x_value": "col_label", "y_value": 0.85}
  ]
}
```

The colorbar can be vertical or horizontal: the tool detects the orientation automatically from the aspect ratio of the selected rectangle.
