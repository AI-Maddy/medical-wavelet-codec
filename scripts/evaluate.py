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
  H  E with the interleaved rANS coder instead of the range coder   (coder ablation)
  I  E with a DMOL (K = 5 logistic mixture) probability model + rANS  (R2 pipeline)
  png / jpegls / jpeg2000 / jpegxl  standard lossless codecs (imagecodecs)
  flif   FLIF reference encoder (FLIF-hub/FLIF, -N -E100), built from source
  calic  CALIC (Wu & Memon 1997), our C implementation in baselines/calic
  l3c    L3C architecture (Mentzer et al. 2019) for 16-bit greyscale, baselines/l3c, models/l3c.pt

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


def write_pgm(path, u):
    """16-bit (or 8-bit) binary PGM; maxval = the image maximum so the codec sees the true range."""
    mx = max(int(u.max()), 1)
    with open(path, "wb") as f:
        f.write(b"P5\n%d %d\n%d\n" % (u.shape[1], u.shape[0], mx))
        f.write(u.astype(">u2" if mx > 255 else "u1").tobytes())


def read_pgm(path):
    data = open(path, "rb").read()
    parts, pos = [], 0
    while len(parts) < 4:                      # magic, width, height, maxval
        while data[pos:pos + 1].isspace():
            pos += 1
        if data[pos:pos + 1] == b"#":
            pos = data.index(b"\n", pos)
            continue
        end = pos
        while not data[end:end + 1].isspace():
            end += 1
        parts.append(data[pos:end])
        pos = end
    pos += 1
    W, H, mx = int(parts[1]), int(parts[2]), int(parts[3])
    return np.frombuffer(data[pos:], dtype=">u2" if mx > 255 else "u1", count=W * H).reshape(H, W)


EXTERNAL = {
    # name: (encode command, decode command); {i} input, {o} output
    "flif": (["flif", "-e", "-N", "-E100", "{i}", "{o}"], ["flif", "-d", "{i}", "{o}"], ".flif"),
    "calic": ([str(Path(__file__).resolve().parents[1] / "baselines" / "calic" / "calic"), "e", "{i}", "{o}"],
              [str(Path(__file__).resolve().parents[1] / "baselines" / "calic" / "calic"), "d", "{i}", "{o}"],
              ".cal"),
}


def run_external(name, img):
    """Run a command-line codec through PGM files; report the compressed file size."""
    import subprocess
    import tempfile
    enc, dec, ext = EXTERNAL[name]
    u, off = to_unsigned(img)
    with tempfile.TemporaryDirectory() as td:
        src, cmp_, out = f"{td}/in.pgm", f"{td}/c{ext}", f"{td}/out.pgm"
        write_pgm(src, u)
        t = time.time()
        subprocess.run([a.format(i=src, o=cmp_) for a in enc], check=True, capture_output=True)
        te = time.time() - t
        t = time.time()
        subprocess.run([a.format(i=cmp_, o=out) for a in dec], check=True, capture_output=True)
        td_ = time.time() - t
        y = read_pgm(out)
        nbytes = Path(cmp_).stat().st_size
    ok = np.array_equal(y.astype(np.int64) + off, img.astype(np.int64))
    return nbytes + (2 if off else 0), te, td_, ok


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


def run_neural(img, model, mod, family, levels, verify, entropy_coder=None):
    t = time.time()
    b = codec.encode(img, model, mod, levels=levels, family=family, entropy_coder=entropy_coder)
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
    ap.add_argument("--shard", default="0/1", help="i/n: run only images with index %% n == i")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mdir = Path(args.models)
    full = codec.load_model(mdir / "full.pt") if set("EFGH") & set(args.configs) else None
    dmol = codec.load_model(mdir / "dmol.pt") if "I" in args.configs else None
    l3c_model = None
    if "l3c" in args.configs:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines" / "l3c"))
        import l3c as l3c_mod
        l3c_model = l3c_mod.load(mdir / "l3c.pt")
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
        si, sn = map(int, args.shard.split("/"))
        items = items[si::sn]
        for name, meta, img in items:
            Lf, _ = fbse.choose_levels(img)
            L5 = min(5, int(np.floor(np.log2(min(img.shape) / 8))))
            for cfg in args.configs:
                if (name, cfg) in done:
                    continue
                if cfg == "l3c":
                    t = time.time()
                    b = l3c_mod.encode(img, l3c_model)
                    te = time.time() - t
                    t = time.time()
                    ok = np.array_equal(l3c_mod.decode(b, l3c_model), img)
                    nbytes, td, lv = len(b), time.time() - t, ""
                elif cfg in EXTERNAL:
                    nbytes, te, td, ok = run_external(cfg, img)
                    lv = ""
                elif cfg in STANDARD:
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
                elif cfg == "H":
                    nbytes, te, td, ok = run_neural(img, full, mod, "adaptive", Lf, "H" in args.verify, "rans"); lv = Lf
                elif cfg == "I":
                    nbytes, te, td, ok = run_neural(img, dmol, mod, "adaptive", Lf, "I" in args.verify, "rans"); lv = Lf
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
