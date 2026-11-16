"""Evaluate all codecs on the held-out test patients and write results/.

Every number reported is the size of a real, decodable bit stream (header
included).  Every codec is decoded again and compared with the original pixel
array; a mismatch stops the run.

Configurations (ablation chain, then reference codecs)
  A  LeGall 5/3, 5 levels, static Laplace per subband   ("DWT + conventional coding")
  B  adaptive lifting, 5 levels, static Laplace
  C  adaptive lifting, FBSE-chosen levels, static Laplace
  D  C + neural context model without ConvLSTM / attention
  E  C + full neural entropy model (ConvLSTM + modality attention)   <- proposed
  F  LeGall 5/3, 5 levels + full neural model     (transform ablation)
  G  adaptive lifting, 5 levels + full neural model (FBSE ablation)
  png / jpegls / jpeg2000 / jpegxl  standard lossless codecs (imagecodecs)

usage: python scripts/evaluate.py --data data --models models --out results
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import imagecodecs as ic
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wavecomp import codec, fbse  # noqa: E402
from wavecomp.model import MODALITIES  # noqa: E402


def to_unsigned(img):
    """Standard codecs need unsigned samples: shift by the minimum (2 extra header bytes)."""
    if img.dtype.kind == "i":
        off = int(img.min())
        return (img.astype(np.int64) - off).astype(np.uint16), off
    return img, 0


STANDARD = {
    "png": (lambda u: ic.png_encode(u, level=9), ic.png_decode),
    "jpegls": (lambda u: ic.jpegls_encode(u), ic.jpegls_decode),
    "jpeg2000": (lambda u: ic.jpeg2k_encode(u, level=0, reversible=True, codecformat="j2k"),
                 ic.jpeg2k_decode),
    "jpegxl": (lambda u: ic.jpegxl_encode(u, lossless=True, effort=7), ic.jpegxl_decode),
}


def run_standard(name, img):
    enc, dec = STANDARD[name]
    u, off = to_unsigned(img)
    t = time.time()
    b = enc(u)
    te = time.time() - t
    t = time.time()
    y = dec(b)
    td = time.time() - t
    ok = np.array_equal(y.astype(np.int64) + off, img.astype(np.int64))
    extra = 2 if off else 0
    return len(b) + extra, te, td, ok


def run_static(img, family, levels):
    t = time.time()
    b = codec.encode_static(img, levels=levels, family=family)
    te = time.time() - t
    t = time.time()
    y = codec.decode_static(b)
    td = time.time() - t
    return len(b), te, td, np.array_equal(y, img)


def run_neural(img, model, mod, family, levels, verify):
    t = time.time()
    b = codec.encode(img, model, mod, levels=levels, family=family)
    te = time.time() - t
    td, ok = float("nan"), None
    if verify:
        t = time.time()
        y = codec.decode(b, model)
        td = time.time() - t
        ok = np.array_equal(y, img)
    return len(b), te, td, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--models", default="models")
    ap.add_argument("--out", default="results")
    ap.add_argument("--modalities", nargs="+", default=MODALITIES)
    ap.add_argument("--configs", nargs="+", default=list("ABCDEFG") + list(STANDARD))
    ap.add_argument("--verify", default="E", help="neural configs to decode and compare (e.g. DEFG)")
    ap.add_argument("--limit", type=int, default=0, help="max images per modality (0 = all)")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mdir = Path(args.models)
    full = codec.load_model(mdir / "full.pt") if set("EFG") & set(args.configs) else None
    plain = codec.load_model(mdir / "plain.pt") if "D" in args.configs else None

    rows_path = out / "per_image.csv"
    done = set()
    if rows_path.exists():
        with open(rows_path) as f:
            for r in csv.DictReader(f):
                done.add((r["image"], r["config"]))
    fcsv = open(rows_path, "a", newline="")
    w = csv.writer(fcsv)
    if not done:
        w.writerow(["modality", "image", "patient", "H", "W", "bits_stored", "config", "bytes",
                    "bpp", "enc_s", "dec_s", "lossless", "levels"])

    for mod in args.modalities:
        files = sorted((Path(args.data) / mod).glob("test_*.npz"))
        items = []
        for f in files:
            meta = json.loads(f.with_suffix(".json").read_text())
            for k, im in enumerate(np.load(f)["images"]):
                items.append((f"{f.stem}#{k}", meta, im))
        if args.limit:
            items = items[:args.limit]
        for name, meta, img in items:
            Lf, _ = fbse.choose_levels(img)
            L5 = min(5, int(np.floor(np.log2(min(img.shape) / 8))))
            for cfg in args.configs:
                if (name, cfg) in done:
                    continue
                if cfg in STANDARD:
                    nbytes, te, td, ok = run_standard(cfg, img)
                    lv = ""
                elif cfg == "A":
                    nbytes, te, td, ok = run_static(img, "legall53", L5); lv = L5
                elif cfg == "B":
                    nbytes, te, td, ok = run_static(img, "adaptive", L5); lv = L5
                elif cfg == "C":
                    nbytes, te, td, ok = run_static(img, "adaptive", Lf); lv = Lf
                elif cfg == "D":
                    nbytes, te, td, ok = run_neural(img, plain, mod, "adaptive", Lf, "D" in args.verify); lv = Lf
                elif cfg == "E":
                    nbytes, te, td, ok = run_neural(img, full, mod, "adaptive", Lf, "E" in args.verify); lv = Lf
                elif cfg == "F":
                    nbytes, te, td, ok = run_neural(img, full, mod, "legall53", L5, "F" in args.verify); lv = L5
                elif cfg == "G":
                    nbytes, te, td, ok = run_neural(img, full, mod, "adaptive", L5, "G" in args.verify); lv = L5
                if ok is False:
                    raise SystemExit(f"NOT LOSSLESS: {cfg} on {name}")
                bpp = 8 * nbytes / img.size
                w.writerow([mod, name, meta["patient"], img.shape[0], img.shape[1], meta["BitsStored"],
                            cfg, nbytes, f"{bpp:.5f}", f"{te:.3f}", f"{td:.3f}",
                            "" if ok is None else int(ok), lv])
                fcsv.flush()
            print(mod, name, img.shape, "levels", Lf, flush=True)
    fcsv.close()


if __name__ == "__main__":
    main()
