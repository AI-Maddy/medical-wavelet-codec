"""Illustrative figures from real test images.

  results/fig_fbse.png      FBSE spectrum, detected boundaries and the chosen depth
  results/fig_lossless.png  original / decoded / difference for one image per modality,
                            decoded from the actual WVC1 stream produced by the full model

usage: python scripts/make_figures.py --data data --models models --out results
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wavecomp import codec, fbse  # noqa: E402

TITLES = {"xray": "Chest X-ray", "ct": "Thoracic CT", "mri": "Brain MRI"}


def first_test_image(data, mod):
    f = sorted((Path(data) / mod).glob("test_*.npz"))[0]
    ims = np.load(f)["images"]
    return ims[len(ims) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--models", default="models")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = Path(args.out)
    imgs = {m: first_test_image(args.data, m) for m in TITLES}

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for ax, (m, im) in zip(axes, imgs.items()):
        f, E = fbse.image_spectrum(im)
        b = fbse.boundaries(f, E)
        L, f1 = fbse.choose_levels(im)
        ax.semilogy(f, E + 1e-9, lw=0.8)
        for x in b:
            ax.axvline(x, color="k", ls=":", lw=0.8)
        ax.axvline(f1, color="r", lw=1.2, label=f"f1 = {f1:.3f}  ->  L = {L}")
        for l in range(1, 7):
            ax.axvline(0.5 / 2 ** l, color="0.7", lw=0.5)
        ax.set_title(f"{TITLES[m]}: FBSE energy spectrum", fontsize=9)
        ax.set_xlabel("normalised frequency (cycles/sample)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_fbse.png", dpi=200)

    model = codec.load_model(Path(args.models) / "full.pt")
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for row, (m, im) in zip(axes, imgs.items()):
        blob = codec.encode(im, model, m)
        rec = codec.decode(blob, model)
        diff = rec.astype(np.int64) - im.astype(np.int64)
        lo, hi = np.percentile(im, [1, 99])
        row[0].imshow(im, cmap="gray", vmin=lo, vmax=hi)
        row[0].set_title(f"{TITLES[m]} original", fontsize=9)
        row[1].imshow(rec, cmap="gray", vmin=lo, vmax=hi)
        row[1].set_title(f"decoded ({8 * len(blob) / im.size:.2f} bpp)", fontsize=9)
        row[2].imshow(diff, cmap="bwr", vmin=-1, vmax=1)
        row[2].set_title(f"difference: max |d| = {np.abs(diff).max()}", fontsize=9)
        for a in row:
            a.axis("off")
    fig.tight_layout()
    fig.savefig(out / "fig_lossless.png", dpi=150)


if __name__ == "__main__":
    main()
