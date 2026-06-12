#!/usr/bin/env python3
"""
Module: experiments/experiment_a/backtest.py
Purpose: Re-run backtest for Experiment A using existing results or retrain.

Usage:
    python experiments/experiment_a/backtest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.experiment_a.train import run_experiment_a

if __name__ == "__main__":
    run_experiment_a()
