"""
Typed tabular row formats for the RMS metric.

Each dataclass represents one data point from a chart JSON, converted to an
explicit named structure.  The ``key()`` method returns the string used for
Hungarian-matching key similarity.

Row types
---------
StandardRow  — bar, line, pie, radar, histogram, heatmap (as special case)
ErrorRow     — errorpoint / errorbar
BoxRow       — boxplot
BubbleRow    — bubble chart (orientation-normalised: always series + cat_label)
ScatterRow   — scatter plot (no categorical label)
MetaRow      — chart-level metadata (e.g. chart_title)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StandardRow:
    series: str
    label: str
    value: float

    def key(self) -> str:
        return self.series + self.label


@dataclass
class ErrorRow:
    series: str
    label: str
    min: float | None = None
    median: float | None = None
    max: float | None = None

    def key(self) -> str:
        return self.series + self.label


@dataclass
class BoxRow:
    series: str
    label: str
    min: float | None = None
    q1: float | None = None
    median: float | None = None
    q3: float | None = None
    max: float | None = None

    def key(self) -> str:
        return self.series + self.label


@dataclass
class BubbleRow:
    """
    Orientation-normalised bubble row.
    ``label``  — value of the categorical axis (x_value or y_value)
    ``value``  — numeric position on the non-categorical axis
    ``z``      — bubble size (optional)
    ``w``      — bubble colour / weight (optional)
    """
    series: str
    label: str
    value: float
    z: float | None = None
    w: float | None = None

    def key(self) -> str:
        return self.series + self.label


@dataclass
class ScatterRow:
    series: str
    x: float
    y: float

    def key(self) -> str:
        return self.series


@dataclass
class MetaRow:
    field: str   # e.g. "chart_title"
    value: str

    def key(self) -> str:
        return "__meta__" + self.field


# Union alias for type hints
ChartRow = StandardRow | ErrorRow | BoxRow | BubbleRow | ScatterRow | MetaRow
