import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import utils


def main():
    # Match the notebook behavior by preferring the active env's TeX tools.
    fmri_bin = os.path.dirname(sys.executable)
    os.environ["PATH"] = fmri_bin + os.pathsep + os.environ["PATH"]

    sns.set(
        context="talk",
        font_scale=1.0,
        color_codes=True,
        palette="deep",
        style="ticks",
        rc={
            "mathtext.fontset": "cm",
            "xtick.direction": "in",
            "ytick.direction": "in",
            "axes.linewidth": 1.5,
            "figure.dpi": 70,
            "text.usetex": True,
            "font.size": 24,
        },
    )

    K = 1000
    Pk = 1.0 / np.arange(1, K + 1)
    Pk = Pk / Pk.sum()

    n = 100000
    samples = np.random.choice(np.arange(1, K + 1), size=n, p=Pk)
    counts = np.bincount(samples, minlength=K + 1)[1:]

    ranks = np.arange(1, K + 1)
    expected = n * Pk
    mask = counts > 0

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.loglog(ranks[mask], counts[mask], ".", ms=3, label="empirical")
    ax.loglog(ranks, expected, "-", lw=2, label="expected")
    ax.set_xlabel("rank k")
    ax.set_ylabel("count")
    ax.set_title("Zipf sampling: empirical vs expected")
    ax.legend()

    savefigdir = "./figs/Figure1"
    out = f"{savefigdir}/Figure1b_zipf_distribution"
    utils.ensure_dir(savefigdir)
    fig.savefig(f"{out}.pdf", format="pdf", bbox_inches="tight")

    pdf_path = f"{out}.pdf"
    print(f"saved={pdf_path}")
    print(f"exists={os.path.exists(pdf_path)}")
    print(f"size={os.path.getsize(pdf_path)}")


if __name__ == "__main__":
    main()
