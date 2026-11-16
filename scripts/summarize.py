"""Turn results/per_image.csv into results/summary.md, results/summary.json and figures.

Rates are pooled over all pixels of a modality (total bits / total pixels).
CR is relative to the 16 bits per pixel the DICOM files allocate.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

NAMES = {
    "A": "A  LeGall 5/3 DWT (5 lv) + static Laplace",
    "B": "B  adaptive lifting (5 lv) + static Laplace",
    "C": "C  adaptive lifting, FBSE levels + static Laplace",
    "D": "D  C + neural context model (no ConvLSTM, no attention)",
    "E": "E  C + ConvLSTM neural entropy model (proposed)",
    "F": "F  LeGall 5/3 (5 lv) + ConvLSTM neural model",
    "G": "G  adaptive lifting (5 lv) + ConvLSTM neural model",
    "png": "PNG (zlib, level 9)",
    "jpegls": "JPEG-LS (CharLS)",
    "jpeg2000": "JPEG 2000 lossless (OpenJPEG, 5/3)",
    "jpegxl": "JPEG XL lossless (libjxl, effort 7)",
}
ORDER = ["png", "jpeg2000", "jpegls", "jpegxl", "A", "B", "C", "D", "E", "F", "G"]
MODS = {"xray": "Chest X-ray (LIDC-IDRI CR/DX)", "ct": "Thoracic CT (LIDC-IDRI)",
        "mri": "Brain MRI (UPENN-GBM)"}


def pooled(d):
    bits = 8 * d.bytes.sum()
    pix = (d.H * d.W).sum()
    bpp = bits / pix
    return dict(bpp=bpp, cr=16 / bpp, n=len(d), enc_s=d.enc_s.mean(),
                dec_s=d.dec_s.mean() if d.dec_s.notna().any() else float("nan"),
                lossless_checked=int(d.lossless.fillna(0).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    res = Path(args.results)
    df = pd.read_csv(res / "per_image.csv")
    summary = {}
    lines = ["# Results (measured)", "",
             "All rates are real bit-stream sizes including headers, pooled over every pixel of the "
             "held-out test patients. CR = 16 / bpp (DICOM stores 16 bits per pixel). "
             "Every stream listed with a check count was decoded and compared bit-for-bit with the original.",
             ""]
    for mod, title in MODS.items():
        d = df[df.modality == mod]
        if d.empty:
            continue
        imgs = d.image.unique()
        # only compare configs on the images every config finished
        done = d.groupby("image").config.nunique()
        full = done[done == done.max()].index
        d = d[d.image.isin(full)]
        bits_stored = ", ".join(str(int(b)) for b in sorted(d.bits_stored.unique()))
        lines += [f"## {title}", "",
                  f"{len(full)} images from {d.patient.nunique()} test patients "
                  f"(of {len(imgs)} attempted), size {int(d.H.median())}x{int(d.W.median())} median, "
                  f"Bits Stored {bits_stored}.", "",
                  "| Codec | bpp | CR | enc s/img | dec s/img | decoded & verified |",
                  "|---|---:|---:|---:|---:|---:|"]
        summary[mod] = {}
        for cfg in ORDER:
            dc = d[d.config == cfg]
            if dc.empty:
                continue
            p = pooled(dc)
            summary[mod][cfg] = p
            dec = "-" if np.isnan(p["dec_s"]) else f"{p['dec_s']:.2f}"
            lines.append(f"| {NAMES[cfg]} | {p['bpp']:.3f} | {p['cr']:.2f} | {p['enc_s']:.2f} | {dec} | "
                         f"{p['lossless_checked']}/{p['n']} |")
        lines.append("")
    (res / "summary.md").write_text("\n".join(lines))
    (res / "summary.json").write_text(json.dumps(summary, indent=1))
    print("\n".join(lines))

    # ablation figure
    cfgs = [c for c in ["A", "B", "C", "D", "E"] if any(c in summary[m] for m in summary)]
    fig, axes = plt.subplots(1, len(summary), figsize=(4.2 * len(summary), 3.4), squeeze=False)
    for ax, (mod, s) in zip(axes[0], summary.items()):
        vals = [s[c]["bpp"] for c in cfgs if c in s]
        ax.bar([c for c in cfgs if c in s], vals, color="#4c72b0")
        for ref, col in [("jpeg2000", "#dd8452"), ("jpegls", "#55a868"), ("jpegxl", "#c44e52")]:
            if ref in s:
                ax.axhline(s[ref]["bpp"], color=col, ls="--", lw=1, label=NAMES[ref].split(" (")[0])
        ax.set_title(MODS[mod], fontsize=9)
        ax.set_ylabel("bits per pixel (lower is better)")
        refs = [s[r]["bpp"] for r in ("jpeg2000", "jpegls", "jpegxl") if r in s]
        ax.set_ylim(0, max(vals + refs) * 1.1)
    axes[0][0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(res / "fig_ablation.png", dpi=200)
    fig.savefig(res / "fig_ablation.pdf")


if __name__ == "__main__":
    main()
