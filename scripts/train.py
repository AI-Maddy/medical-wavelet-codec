"""Train the neural entropy model on random crops of the *training* patients.

The loss is the exact code length the entropy coder will spend (bits per pixel
under the discretised Laplace model, or under the DMOL model with --mixture K),
so training directly minimises file size.

usage: python scripts/train.py --data data --out models/full.pt [--no-lstm] [--no-attention] [--mixture 5]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wavecomp import codec, lifting  # noqa: E402
from wavecomp.model import MODALITIES, EntropyModel  # noqa: E402


def load_split(data, mod, split):
    out = []
    for f in sorted((Path(data) / mod).glob(f"{split}_*.npz")):
        for im in np.load(f)["images"]:
            out.append(codec.pack(im)[0])     # same histogram packing as the codec
    return out


def make_bank(images, n, crop, levels, rng):
    bank = []
    tries = 0
    while len(bank) < n and tries < 50 * n:
        tries += 1
        im = images[rng.integers(len(images))]
        if min(im.shape) < crop:
            continue
        y = rng.integers(0, im.shape[0] - crop + 1)
        x = rng.integers(0, im.shape[1] - crop + 1)
        c = im[y:y + crop, x:x + crop].astype(np.int64)
        if c.std() < 1 and rng.random() < 0.9:     # mostly skip pure background
            continue
        if rng.random() < 0.5:
            c = c[:, ::-1]
        L = levels[rng.integers(len(levels))]
        d = lifting.forward(np.ascontiguousarray(c), L, "adaptive", 4)
        bands = codec._bands_coarse_to_fine(d)
        stats = codec.all_stats(d)
        # intermediate low bands LL_{L-1} .. LL_1 (what the decoder reconstructs)
        lows = []
        cur = d.ll
        for li, l in enumerate(range(L - 1, 0, -1)):
            cur = lifting.inverse_level(cur, d.bands[l], d.filters[l])
            lows.append(cur)
        bank.append(dict(L=L, bands=[b.astype(np.int32) for b in bands], stats=stats,
                         lows=[x.astype(np.int32) for x in lows]))
    return bank


def batch_from(bank_items, mod_idx):
    B = len(bank_items)
    nb = len(bank_items[0]["bands"])
    tb = [torch.from_numpy(np.stack([it["bands"][i] for it in bank_items]).astype(np.int64))[:, None]
          for i in range(nb)]
    stats = []
    for i in range(nb):
        s = np.array([it["stats"][i] for it in bank_items], dtype=np.float64)
        stats.append((s[:, 0], s[:, 1], s[:, 2], s[:, 3]))
    lows = [torch.from_numpy(np.stack([it["lows"][j] for it in bank_items]).astype(np.int64))[:, None]
            for j in range(len(bank_items[0]["lows"]))]
    H, W = bank_items[0]["bands"][0].shape
    L = bank_items[0]["L"]
    shapes = [(H << i, W << i) for i in range(L)]
    return tb, stats, lows, shapes, torch.full((B,), mod_idx, dtype=torch.long)


def run_batch(model, items, mod_idx):
    tb, stats, lows, shapes, mod = batch_from(items, mod_idx)
    coder = codec.TrainCoder()
    codec.traverse(model, coder, mod, stats, shapes, tb, lambda li, ll, three: lows[li])
    npix = len(items) * (shapes[-1][0] * 2) * (shapes[-1][1] * 2)
    return coder.bits / npix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="models/full.pt")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--bank", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--no-lstm", action="store_true")
    ap.add_argument("--no-attention", action="store_true")
    ap.add_argument("--mixture", type=int, default=0,
                    help="K > 0: discretised mixture of K logistics (DMOL) instead of Laplace")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--modalities", nargs="+", default=MODALITIES)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    banks, vals = {}, {}
    for mod in args.modalities:
        imgs = load_split(args.data, mod, "train")
        t = time.time()
        banks[mod] = make_bank(imgs, args.bank, args.crop, [2, 3], rng)
        vals[mod] = make_bank(imgs, 64, args.crop, [3], np.random.default_rng(123))
        print(f"{mod}: {len(imgs)} train images -> {len(banks[mod])} crops ({time.time() - t:.0f}s)",
              flush=True)

    model = EntropyModel(use_lstm=not args.no_lstm, use_attention=not args.no_attention,
                         mixture=args.mixture)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps,
                                                pct_start=0.05)
    log = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        mod = args.modalities[int(rng.integers(len(args.modalities)))]
        mi = MODALITIES.index(mod)
        L = int(rng.choice([2, 3]))
        pool = [it for it in banks[mod] if it["L"] == L]
        items = [pool[i] for i in rng.integers(0, len(pool), args.batch)]
        loss = run_batch(model, items, mi)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 100 == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                v = {m: float(np.mean([run_batch(model, vals[m][i:i + 16], MODALITIES.index(m)).item()
                                       for i in range(0, len(vals[m]), 16)])) for m in args.modalities}
            model.train()
            rec = dict(step=step, loss=float(loss), val_bpp=v, minutes=(time.time() - t0) / 60)
            log.append(rec)
            print(json.dumps(rec), flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(cfg=model.cfg, state=model.state_dict(), args=vars(args), log=log), out)
    print("saved", out)


if __name__ == "__main__":
    main()
