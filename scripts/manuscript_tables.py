"""Measured versions of the manuscript's result tables and figures (R2 numbering).

Reads only the measured per-image results written by evaluate.py
(results/per_image.csv) and writes, to results/manuscript/:

  table3_bpp_cr.md      Table 3  bpp per dataset, average bpp, average CR
  table4_ablation.md    Table 4  component ablation (CT, plus the other two modalities)
  table5_throughput.md  Table 5  encode / decode throughput in MP/s
  table6_stats.md       Table 6  per-image statistics and paired tests against the proposed codec
  fig4_avg_bpp.png/pdf  Figure 4 average bpp per method
  fig5_avg_cr.png/pdf   Figure 5 average CR per method
  fig6_bpp_by_dataset   Figure 6 bpp per method and dataset
  fig7_cr_by_dataset    Figure 7 CR per method and dataset
  manuscript_numbers.json  every number above, machine readable

Nothing here is typed in by hand: all values come from real, decoded bit streams.
"Proposed" is reported twice: E (Laplace + range coder) and I (DMOL + rANS, the
R2 pipeline); Table 6 pairs every codec against I. CALIC is our C implementation (baselines/calic), FLIF the reference encoder, and L3C
our 16-bit greyscale reimplementation trained on the training patients (baselines/l3c).
JPEG XL is included as the current standard.

Two compression-ratio conventions are written, because the manuscript uses the second:
  CR16     = 16 / bpp                     (DICOM stores every pixel in 16 bits)
  CRstored = Bits Stored / bpp per image  (the image's own bit depth, then averaged)

Usage: python scripts/manuscript_tables.py --results results
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

METHODS = [("png", "PNG"), ("jpegls", "JPEG-LS"), ("jpeg2000", "JPEG 2000-LL"),
           ("calic", "CALIC"), ("flif", "FLIF"), ("l3c", "L3C (16-bit reimpl., 150k GPU steps)"), ("jpegxl", "JPEG XL"), ("E", "Proposed (Laplace + range coder)"),
           ("I", "Proposed (DMOL + rANS)")]
ABLATION = [("A", "Fixed LeGall 5/3 DWT + static Laplace coding"),
            ("B", "Adaptive lifting (ALDWT)"),
            ("C", "ALDWT + FBSE-chosen depth"),
            ("D", "C + neural context model (no ConvLSTM, no attention)"),
            ("E", "C + ConvLSTM-NEM with modality attention, Laplace + range coder"),
            ("H", "E with rANS instead of the range coder"),
            ("I", "E with DMOL (K = 5) + rANS (full R2 pipeline)"),
            ("F", "E with fixed LeGall 5/3"),
            ("G", "E with fixed 5 levels")]
REF = "I"      # the configuration the statistics are paired against
DATASETS = [("xray", "Chest X-ray"), ("ct", "Thoracic CT"), ("mri", "Brain MRI")]


def pooled(df):
    """bpp over all pixels, CR16, mean of per-image stored-depth CR, MP/s."""
    px = (df.H * df.W).sum()
    bpp = 8.0 * df.bytes.sum() / px
    return dict(bpp=bpp, cr16=16.0 / bpp,
                cr_stored=float((df.bits_stored / df.bpp).mean()),
                enc_mps=px / 1e6 / df.enc_s.sum(), dec_mps=px / 1e6 / df.dec_s.sum(),
                n=len(df), lossless=int(df.lossless.sum()))


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def paired_tests(a, b):
    from scipy import stats
    d = a - b
    t = stats.ttest_rel(a, b).pvalue
    w = stats.wilcoxon(a, b).pvalue if np.any(d != 0) else 1.0
    return t, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    df = pd.read_csv(os.path.join(args.results, "per_image.csv"))
    out = os.path.join(args.results, "manuscript")
    os.makedirs(out, exist_ok=True)

    S = {m: {c: pooled(g) for c, g in dm.groupby("config")} for m, dm in df.groupby("modality")}
    global METHODS
    METHODS = [(c, n) for c, n in METHODS if all(c in S[m] for m, _ in DATASETS)]
    numbers = {"per_dataset": S}

    # Table 3
    rows, avg = [], {}
    for c, name in METHODS:
        b = [S[m][c]["bpp"] for m, _ in DATASETS]
        cr16 = np.mean([S[m][c]["cr16"] for m, _ in DATASETS])
        crs = np.mean([S[m][c]["cr_stored"] for m, _ in DATASETS])
        avg[c] = dict(bpp=float(np.mean(b)), cr16=float(cr16), cr_stored=float(crs))
        rows.append([name] + [f"{x:.2f}" for x in b] + [f"{np.mean(b):.2f}", f"{cr16:.2f}x", f"{crs:.2f}x"])
    numbers["average"] = avg
    hdr = ["Method"] + [f"{n} (bpp)" for _, n in DATASETS] + ["Average bpp", "Avg CR16", "Avg CR (stored depth)"]
    note = ("Measured on held-out test patients; every stream decoded and checked bit-exact. "
            "Average = arithmetic mean of the three dataset values.\n\n")
    open(os.path.join(out, "table3_bpp_cr.md"), "w").write("# Table 3 (measured)\n\n" + note + md_table(hdr, rows))

    # Table 4
    txt = "# Table 4 (measured ablation)\n\n"
    for m, name in [("ct", "Thoracic CT (the dataset of manuscript Table 4)"), ("xray", "Chest X-ray"), ("mri", "Brain MRI")]:
        r = [[c, d, f"{S[m][c]['bpp']:.3f}", f"{S[m][c]['cr16']:.2f}x", f"{S[m][c]['cr_stored']:.2f}x"] for c, d in ABLATION]
        txt += f"## {name}\n\n" + md_table(["Config", "Components", "bpp", "CR16", "CR (stored depth)"], r) + "\n"
    txt += ("A-H use a discretised Laplace probability model; H swaps the range coder for rANS and I "
            "swaps the Laplace for a K = 5 discretised mixture of logistics (DMOL) as well.\n")
    open(os.path.join(out, "table4_ablation.md"), "w").write(txt)

    # Table 5
    rows = []
    for c, name in METHODS:
        px = df[df.config == c]
        tot = (px.H * px.W).sum() / 1e6
        rows.append([name, f"{tot / px.enc_s.sum():.2f}", f"{tot / px.dec_s.sum():.2f}"])
    open(os.path.join(out, "table5_throughput.md"), "w").write(
        "# Table 5 (measured throughput, 4-core CPU, no GPU, single process)\n\nMP/s over all test pixels of the three datasets.\n\n"
        + md_table(["Method", "Encode (MP/s)", "Decode (MP/s)"], rows))

    # Table 6: per-image statistics, paired against the proposed codec
    rows = []
    stats6 = {}
    for m, mname in DATASETS:
        dm = df[df.modality == m].pivot_table(index="image", columns="config", values="bpp")
        for c, name in METHODS:
            x = dm[c].values
            mean, sd = x.mean(), x.std(ddof=1)
            from scipy import stats
            h = stats.t.ppf(0.975, len(x) - 1) * sd / np.sqrt(len(x))
            if c == REF:
                pt = pw = None
                rows.append([mname, name, len(x), f"{mean:.3f} ± {sd:.3f}", f"[{mean - h:.3f}, {mean + h:.3f}]", "—", "—", "—"])
            else:
                pt, pw = paired_tests(x, dm[REF].values)
                win = (dm[REF].values < x).mean()
                rows.append([mname, name, len(x), f"{mean:.3f} ± {sd:.3f}", f"[{mean - h:.3f}, {mean + h:.3f}]",
                             f"{pt:.2g}", f"{pw:.2g}", f"{100 * win:.0f}%"])
            stats6[f"{m}/{c}"] = dict(mean=float(mean), sd=float(sd), ci=[float(mean - h), float(mean + h)],
                                      p_t=None if pt is None else float(pt), p_wilcoxon=None if pw is None else float(pw))
    numbers["table6"] = stats6
    open(os.path.join(out, "table6_stats.md"), "w").write(
        "# Table 6 (measured, per image)\n\nMean ± SD of per-image bpp, 95% t-interval of the mean, and paired tests of each "
        "codec against the proposed codec on the same images. 'Proposed better' = share of images where the proposed "
        "stream is smaller. Images within a patient are not independent, so treat p-values as descriptive.\n\n"
        + md_table(["Dataset", "Method", "n", "bpp mean ± SD", "95% CI", "p (paired t)", "p (Wilcoxon)", "Proposed better"], rows))

    json.dump(numbers, open(os.path.join(out, "manuscript_numbers.json"), "w"), indent=1)

    # Figures 4-7
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [n.replace(" (", "\n(") for _, n in METHODS]
    colors = ["#9aa5b1"] * (len(METHODS) - 2) + ["#7fa7e8", "#2a6fdb"]

    def bar(vals, ylabel, fname, fmt):
        fig, ax = plt.subplots(figsize=(9, 3.9))
        bars = ax.bar(names, vals, color=colors)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v), ha="center", va="bottom", fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(out, f"{fname}.{ext}"), dpi=200)
        plt.close(fig)

    def grouped(key, ylabel, fname):
        fig, ax = plt.subplots(figsize=(9, 4))
        w = 0.8 / len(METHODS)
        x = np.arange(len(DATASETS))
        for i, (c, n) in enumerate(METHODS):
            ax.bar(x + (i - (len(METHODS) - 1) / 2) * w, [S[m][c][key] for m, _ in DATASETS], w, label=n,
                   color=plt.cm.viridis(i / (len(METHODS) - 1)))
        ax.set_xticks(x, [n for _, n in DATASETS])
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8, ncol=3, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(out, f"{fname}.{ext}"), dpi=200)
        plt.close(fig)

    bar([avg[c]["bpp"] for c, _ in METHODS], "Average bits per pixel", "fig4_avg_bpp", "{:.2f}")
    bar([avg[c]["cr_stored"] for c, _ in METHODS], "Average compression ratio (stored depth)", "fig5_avg_cr", "{:.2f}x")
    grouped("bpp", "Bits per pixel", "fig6_bpp_by_dataset")
    grouped("cr_stored", "Compression ratio (stored depth)", "fig7_cr_by_dataset")
    print("wrote", out)


if __name__ == "__main__":
    main()
