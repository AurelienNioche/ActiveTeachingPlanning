import os

import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = "fig"
os.makedirs(FIG_DIR, exist_ok=True)


def plot_loss(hist_loss, zoom_start=1000):
    fig, axes = plt.subplots(ncols=2, figsize=(10, 5))
    ax = axes[0]
    ax.plot(hist_loss)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")

    ax = axes[1]
    ax.plot(np.arange(len(hist_loss))[-zoom_start:], hist_loss[-zoom_start:])
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")

    fig.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"inference_loss.pdf"))
