"""Plot log10 training loss per model, one panel per experiment regime."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nca import EXPERIMENTS

ROOT = Path(__file__).resolve().parent.parent


def smooth(v: np.ndarray, k: int = 50) -> np.ndarray:
    return np.convolve(v, np.ones(k) / k, mode="valid") if len(v) > k else v


def main():
    fig, axes = plt.subplots(1, len(EXPERIMENTS), figsize=(15, 4), sharey=True)
    for ax, exp in zip(axes, EXPERIMENTS):
        for f in sorted((ROOT / "models" / "runs").glob(f"*_{exp}.loss.json")):
            loss = np.array(json.loads(f.read_text()))
            ax.plot(smooth(np.log10(loss)), label=f.name.split("_")[0], lw=1)
        ax.set_title(exp)
        ax.set_xlabel("training step")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("log10 MSE (50-step mean)")
    axes[-1].legend(fontsize=7, ncol=2)
    out = ROOT / "eval" / "loss_curves.png"
    out.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(out)


if __name__ == "__main__":
    main()
