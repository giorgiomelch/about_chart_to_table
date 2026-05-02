#!/bin/bash
# Generate synthetic charts for all 9 types.
# Run from the project root: bash src/chart_factory/_create_all_charts.sh

NUM_CHARTS=50

python src/chart_factory/generate_all.py --n "$NUM_CHARTS" --types all
