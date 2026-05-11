"""
Relative Mapping Similarity (RMS) metric package.

Public API
----------
compute_rms(predicted, target, chart_type, ...) -> dict

Types re-exported for downstream consumers:
    AxisRanges
    StandardRow, ErrorRow, BoxRow, BubbleRow, ScatterRow, MetaRow
"""

from .core import compute_rms
from .types import AxisRanges
from ..row_types import (
    BoxRow, BubbleRow, ErrorRow, MetaRow, ScatterRow, StandardRow,
)

__all__ = [
    "compute_rms",
    "AxisRanges",
    "StandardRow",
    "ErrorRow",
    "BoxRow",
    "BubbleRow",
    "ScatterRow",
    "MetaRow",
]
