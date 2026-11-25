"""Shared paths and helpers for the whole project."""
import os
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS = ROOT / "results"
FIG_DIR = RESULTS / "figures"
MODEL_DIR = RESULTS / "models"
for _d in (DATA_DIR, RESULTS, FIG_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

PED2_DIR = DATA_DIR / "UCSD_Anomaly_Dataset.v1p2" / "UCSDped2"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def plot_style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    })
    return plt
