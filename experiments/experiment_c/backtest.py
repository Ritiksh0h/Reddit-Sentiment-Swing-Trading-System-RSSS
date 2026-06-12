#!/usr/bin/env python3
"""
Module: experiments/experiment_c/backtest.py
Purpose: Re-run backtest for Experiment C using existing results or retrain.

# WAITING FOR EXPANDED DATA — see train.py for details.

Usage:
    python experiments/experiment_c/backtest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.experiment_c.train import run_experiment_c

if __name__ == "__main__":
    run_experiment_c()
