"""L3C baseline for 8- to 16-bit greyscale images.

L3C (Mentzer, Agustsson, Tschannen, Timofte, Van Gool, "Practical Full Resolution
Learned Lossless Image Compression", CVPR 2019) is a hierarchical, non-autoregressive
model. The published code and weights (github.com/fab-jul/L3C-PyTorch, GPL-3) handle
8-bit RGB only: the pixel model is a 256-level logistic mixture with RGB channel
coupling, and the arithmetic coder needs a full CDF table per pixel. This file is an
independent reimplementation of the same architecture for one channel and any bit
depth, written from the paper (no code copied):

  * S = 3 scales. Encoder E(s): stride-2 conv, 8 residual blocks (Cf = 64), conv to
    C = 5 channels, quantised to L = 25 levels in [-1, 1] (soft-to-hard quantisation,
    sigma = 2, straight-through hard values). E(s) is fed the features of E(s-1).
  * Decoder D(s): z(s), fused with the features of D(s+1), 8 residual blocks,
    conv + pixel shuffle back to the resolution of scale s-1.
  * Probability classifier P(s): atrous convolutions (dilation 1, 2, 4), then a
    discretised mixture of K logistics: K = 10 for the image, K = 10 for each latent.
    z(3) is coded with a uniform prior.
  * Every pixel of a scale is coded in parallel (L3C's speed property).

Adaptations for medical images (needed, not part of L3C): the image is normalised by its
own mean and standard deviation (8 bytes in the header) instead of the fixed RGB mean,
and the pixel alphabet is the image's [min, max] (8 bytes). Coding uses our interleaved
rANS (wavecomp/rans.py), which works with any alphabet size.

usage:
  python baselines/l3c/l3c.py train --data data --out models/l3c.pt [--steps 2500]
  On a GPU (used automatically when available) a fair run is about
  --steps 150000 --batch 16 --lr 1e-4; coding/evaluation stays on the CPU.
  --ckpt PATH keeps the full training state (optimiser, LR schedule, step, RNG) and resumes
  from it, e.g. after a Colab disconnect (notebooks/train_l3c_colab.ipynb).
  (evaluation: scripts/evaluate.py --configs l3c)
"""
import argparse
import io
import json
import struct
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from wavecomp import codec, rans  # noqa: E402

S, C, L, CF, NBLK, K = 3, 5, 25, 64, 8, 10
LEVELS = torch.linspace(-1, 1, L)
MAGIC = b"L3C1"


class Res(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.a = nn.Conv2d(c, c, 3, padding=1)
        self.b = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.b(F.relu(self.a(x)))


class Enc(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.down = nn.Conv2d(cin, CF, 5, stride=2, padding=2)
        self.body = nn.Sequential(*[Res(CF) for _ in range(NBLK)])
        self.out = nn.Conv2d(CF, C, 1)

    def forward(self, x):
        f = self.down(x)
        f = f + self.body(f)
        return f, self.out(f)


class Dec(nn.Module):
    def __init__(self, fuse):
        super().__init__()
        self.inp = nn.Conv2d(C, CF, 1)
        self.fuse = nn.Conv2d(2 * CF, CF, 1) if fuse else None
        self.body = nn.Sequential(*[Res(CF) for _ in range(NBLK)])
        self.up = nn.Conv2d(CF, 4 * CF, 3, padding=1)

    def features(self, z, f_above=None):
        f = self.inp(z)
        if self.fuse is not None:
            f = self.fuse(torch.cat([f, f_above], 1))
        return f + self.body(f)

    def forward(self, z, f_above=None):
        return F.pixel_shuffle(self.up(self.features(z, f_above)), 2)


class ProbClf(nn.Module):
    """Atrous probability classifier -> DMOL parameters (logits, means, log-scales)."""
    def __init__(self, cout):
        super().__init__()
        self.cout = cout
        self.br = nn.ModuleList([nn.Conv2d(CF, CF, 3, padding=d, dilation=d) for d in (1, 2, 4)])
        self.out = nn.Conv2d(3 * CF, cout * 3 * K, 1)

    def forward(self, f):
        h = self.out(torch.cat([F.relu(b(f)) for b in self.br], 1))
        B, _, H, W = h.shape
        h = h.view(B, self.cout, 3 * K, H, W)
        return h[:, :, :K], h[:, :, K:2 * K], h[:, :, 2 * K:].clamp(-7, 7)


def quantise(z, sigma=2.0):
    lv = LEVELS.to(z.device)
    d = (z.unsqueeze(-1) - lv) ** 2
    soft = (torch.softmax(-sigma * d, -1) * lv).sum(-1)
    idx = d.argmin(-1)
    hard = lv[idx]
    return soft + (hard - soft).detach(), idx


class L3C(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.ModuleList([Enc(1)] + [Enc(CF) for _ in range(S - 1)])
        self.dec = nn.ModuleList([Dec(fuse=s < S - 1) for s in range(S)])
        self.prob = nn.ModuleList([ProbClf(1)] + [ProbClf(C) for _ in range(S - 1)])

    def encode_latents(self, xn):
        zs, idxs, f = [], [], xn
        for s in range(S):
            f, z = self.enc[s](f)
            zq, idx = quantise(z)
            zs.append(zq)
            idxs.append(idx)
        return zs, idxs

    def image_prediction(self, z0, f_above, rows=64):
        """P[0] computed in horizontal strips (exact same arithmetic in encoder and decoder),
        so full-resolution radiographs fit in memory. Halo: 1 row for the 3x3 up-conv and
        2 rows (4 after upsampling) for the dilated classifier."""
        g = self.dec[0].features(z0, f_above)
        H = g.shape[2]
        if not torch.is_grad_enabled() and H > rows:
            outs = []
            for r0 in range(0, H, rows):
                a, b = max(0, r0 - 3), min(H, r0 + rows + 3)
                up = F.pixel_shuffle(self.dec[0].up(g[:, :, a:b]), 2)
                Pp = self.prob[0](up)
                lo_, hi_ = 2 * (r0 - a), 2 * (r0 - a) + 2 * (min(H, r0 + rows) - r0)
                outs.append(tuple(t[..., lo_:hi_, :] for t in Pp))
            return tuple(torch.cat([o[i] for o in outs], -2) for i in range(3))
        return self.prob[0](F.pixel_shuffle(self.dec[0].up(g), 2))

    def predictions(self, zs):
        """Top-down: returns P(s) for s = 0..S-1 (P[0] predicts the image, P[s] predicts z(s-1))."""
        P = [None] * S
        f_above = None
        for s in range(S - 1, 0, -1):
            f = self.dec[s](zs[s], f_above)
            P[s] = self.prob[s](f)
            f_above = f
        P[0] = self.image_prediction(zs[0], f_above)
        return P


# ---------------------------------------------------------------------------
# losses (bits)
# ---------------------------------------------------------------------------
def mix_bits(x, P, lo, hi):
    """x: (B,c,H,W) integers; P = (logits, mu, scale) each (B,c,K,H,W) in x units."""
    lg, mu, sc = P
    B, c = x.shape[:2]
    flat = lambda t: t.permute(0, 1, 3, 4, 2).reshape(B * c, -1, t.shape[2]).permute(0, 2, 1)[..., None]  # noqa
    xx = x.reshape(B * c, 1, -1)[..., None].float()
    lo_ = np.repeat(np.asarray(lo, dtype=np.float64), c) if np.ndim(lo) else np.full(B * c, lo, dtype=np.float64)
    hi_ = np.repeat(np.asarray(hi, dtype=np.float64), c) if np.ndim(hi) else np.full(B * c, hi, dtype=np.float64)
    if x.device.type == "cpu":
        return codec.dmol_nll_bits(xx, flat(lg), flat(mu), flat(sc), lo_, hi_).sum()
    return _dmol_bits_dev(xx, flat(lg), flat(mu), flat(sc), lo_, hi_).sum()


def _dmol_bits_dev(x, logits, mu, scale, lo, hi):
    """Same as codec.dmol_nll_bits, with lo/hi on the tensors' device (GPU training)."""
    lo = torch.as_tensor(lo, dtype=x.dtype, device=x.device).view(-1, 1, 1, 1)
    hi = torch.as_tensor(hi, dtype=x.dtype, device=x.device).view(-1, 1, 1, 1)
    inv = 1.0 / scale
    c = x - mu
    up, dn, mid = inv * (c + 0.5), inv * (c - 0.5), inv * c
    log_cdf_up = up - F.softplus(up)
    log_1m_cdf_dn = -F.softplus(dn)
    diff = torch.sigmoid(up) - torch.sigmoid(dn)
    log_pdf_mid = mid - torch.log(scale) - 2.0 * F.softplus(mid)
    log_mid = torch.where(diff > 1e-5, torch.log(torch.clamp(diff, min=1e-12)), log_pdf_mid)
    logp = torch.where(x <= lo, log_cdf_up, torch.where(x >= hi, log_1m_cdf_dn, log_mid))
    logp = torch.logsumexp(logp + torch.log_softmax(logits, 1), 1, keepdim=True)
    return -torch.clamp(logp, min=-31 * np.log(2)) / np.log(2)


def image_params(P0, m, sd):
    lg, mu, ls = P0
    m = m.view(-1, 1, 1, 1, 1)
    sd = sd.view(-1, 1, 1, 1, 1)
    return lg, mu * sd + m, sd * torch.exp(ls)


def latent_params(Ps):
    lg, mu, ls = Ps                        # index units: mean in [-1, 1] mapped to 0..L-1
    return lg, (mu + 1) * (L - 1) / 2, torch.exp(ls) * (L - 1) / 2


def total_bits(model, x):
    """x: (B,1,H,W) int64, H and W multiples of 8."""
    xf = x.float()
    m = xf.mean((1, 2, 3))
    sd = xf.std((1, 2, 3)) + 1.0
    xn = (xf - m.view(-1, 1, 1, 1)) / sd.view(-1, 1, 1, 1)
    zs, idxs = model.encode_latents(xn)
    P = model.predictions(zs)
    lo = x.amin((1, 2, 3)).cpu().numpy()
    hi = x.amax((1, 2, 3)).cpu().numpy()
    bits = mix_bits(x, image_params(P[0], m, sd), lo, hi)
    for s in range(1, S):
        bits = bits + mix_bits(idxs[s - 1], latent_params(P[s]), 0, L - 1)
    bits = bits + idxs[-1].numel() * np.log2(L)
    return bits


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def load_split(data, mod, split):
    out = []
    for f in sorted((Path(data) / mod).glob(f"{split}_*.npz")):
        out.extend(np.asarray(im, dtype=np.int64) for im in np.load(f)["images"])
    return out


def train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    imgs = {m: [im for im in load_split(args.data, m, "train") if min(im.shape) >= args.crop]
            for m in ["xray", "ct", "mri"]}
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", dev, flush=True)
    model = L3C()
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location="cpu", weights_only=False)["state"])
    model = model.to(dev)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False      # keep fp32 training numerics
        torch.backends.cudnn.allow_tf32 = False
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    log, t0, start = [], time.time(), 1
    ckpt = Path(args.ckpt) if args.ckpt else None
    if ckpt and ckpt.exists():
        # resume after a disconnect: model, optimiser, LR schedule, step and RNG state
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        rng.bit_generator.state = ck["np_rng"]
        torch.set_rng_state(ck["torch_rng"])
        log, start = ck["log"], ck["step"] + 1
        t0 -= 60 * (log[-1]["minutes"] if log else 0)
        print(f"resumed from {ckpt} at step {ck['step']}", flush=True)
    for step in range(start, args.steps + 1):
        mod = ["xray", "ct", "mri"][int(rng.integers(3))]
        batch = []
        while len(batch) < args.batch:
            im = imgs[mod][rng.integers(len(imgs[mod]))]
            y = rng.integers(0, im.shape[0] - args.crop + 1)
            x = rng.integers(0, im.shape[1] - args.crop + 1)
            c = im[y:y + args.crop, x:x + args.crop]
            if c.std() < 1 and rng.random() < 0.9:
                continue
            batch.append(c[:, ::-1] if rng.random() < 0.5 else c)
        x = torch.from_numpy(np.ascontiguousarray(np.stack(batch)))[:, None].to(dev)
        loss = total_bits(model, x) / x.numel()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 50 == 0 or step == args.steps:
            rec = dict(step=step, mod=mod, bpp=float(loss), minutes=(time.time() - t0) / 60)
            log.append(rec)
            print(json.dumps(rec), flush=True)
        if step % args.save_every == 0 or step == args.steps:
            state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            _atomic_save(dict(state=state, args=vars(args), log=log, step=step), args.out)
            if ckpt:
                _atomic_save(dict(state=state, opt=opt.state_dict(), sched=sched.state_dict(),
                                  np_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                                  args=vars(args), log=log, step=step), ckpt)
    print("saved", args.out)


def _atomic_save(obj, path):
    """Write to a temporary file, then rename, so a disconnect never leaves a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# real coding with rANS
# ---------------------------------------------------------------------------
def _pad(img):
    H, W = img.shape
    ph, pw = (-H) % 8, (-W) % 8
    return np.pad(img, ((0, ph), (0, pw)), mode="edge")


def _cdf(P, c, lo, hi):
    lg, mu, sc = (t[0, c].detach().numpy().reshape(K, -1).T for t in P)
    lg, mu, sc = codec.quantise_mixture(lg, mu, sc, lo, hi)
    return rans.logistic_mixture_cdf(lg, mu, sc)


def _uniform(lo, hi):
    def F(y, idx):
        return (y - lo + 0.5) / (hi - lo + 1)
    return F


def _params_from_header(m, sd):
    return torch.tensor([m], dtype=torch.float32), torch.tensor([sd], dtype=torch.float32)


@torch.no_grad()
def encode(img, model):
    img = np.asarray(img)
    H, W = img.shape
    x = torch.from_numpy(_pad(img).astype(np.int64))[None, None]
    m, sd = float(np.float32(x.float().mean())), float(np.float32(x.float().std() + 1.0))
    mt, st = _params_from_header(m, sd)
    lo, hi = int(x.min()), int(x.max())
    xn = (x.float() - mt) / st
    _, idxs = model.eval().encode_latents(xn)
    zs = [LEVELS[i] for i in idxs]       # exact level values, as the decoder sees them
    P = model.predictions(zs)
    enc = rans.RansEncoder(rans.lanes_for(img.size))
    top = idxs[-1][0].numpy()
    enc.add(top.ravel(), _uniform(0, L - 1), 0, L - 1)
    for s in range(S - 1, 0, -1):             # z(s-1) given P[s]
        Pz = latent_params(P[s])
        for c in range(C):
            enc.add(idxs[s - 1][0, c].numpy().ravel(), _cdf(Pz, c, 0, L - 1), 0, L - 1)
    if hi > lo:
        enc.add(x[0, 0].numpy().ravel(), _cdf(image_params(P[0], mt, st), 0, lo, hi), lo, hi)
    f = io.BytesIO()
    f.write(MAGIC)
    f.write(struct.pack("<IIBffii", H, W, ["uint8", "int8", "uint16", "int16", "uint32", "int32"].index(str(img.dtype)),
                        m, sd, lo, hi))
    f.write(struct.pack("<I", zlib.crc32(img.tobytes())))
    f.write(enc.finish())
    return f.getvalue()


@torch.no_grad()
def decode(blob, model):
    assert blob[:4] == MAGIC
    H, W, dt, m, sd, lo, hi = struct.unpack_from("<IIBffii", blob, 4)
    crc, = struct.unpack_from("<I", blob, 4 + 25)
    dec = rans.RansDecoder(blob[4 + 29:])
    Hp, Wp = H + (-H) % 8, W + (-W) % 8
    shapes = [(Hp >> (s + 1), Wp >> (s + 1)) for s in range(S)]
    mt, st = _params_from_header(m, sd)
    model.eval()
    idxs = [None] * S
    n = shapes[-1][0] * shapes[-1][1] * C
    idxs[-1] = torch.from_numpy(dec.decode(n, _uniform(0, L - 1), 0, L - 1)).view(1, C, *shapes[-1])
    zs = [None] * S
    zs[-1] = LEVELS[idxs[-1]]
    f_above = None
    for s in range(S - 1, -1, -1):
        if s > 0:
            f = model.dec[s](zs[s], f_above)
            Ps = model.prob[s](f)
            f_above = f
        else:
            Ps = model.image_prediction(zs[0], f_above)
        if s > 0:
            Pz = latent_params(Ps)
            h, w = shapes[s - 1]
            planes = [torch.from_numpy(dec.decode(h * w, _cdf(Pz, c, 0, L - 1), 0, L - 1)).view(h, w)
                      for c in range(C)]
            idxs[s - 1] = torch.stack(planes)[None]
            zs[s - 1] = LEVELS[idxs[s - 1]]
        else:
            if hi > lo:
                v = dec.decode(Hp * Wp, _cdf(image_params(Ps, mt, st), 0, lo, hi), lo, hi)
            else:
                v = np.full(Hp * Wp, lo, dtype=np.int64)
    out = v.reshape(Hp, Wp)[:H, :W].astype(["uint8", "int8", "uint16", "int16", "uint32", "int32"][dt])
    if zlib.crc32(out.tobytes()) != crc or not dec.check_final():
        raise ValueError("L3C decode mismatch")
    return out


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = L3C()
    m.load_state_dict(ck["state"])
    return m.eval()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="models/l3c.pt")
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default="", help="continue from this checkpoint (new LR schedule)")
    ap.add_argument("--ckpt", default="", help="full training state; resumed from if it exists")
    ap.add_argument("--save-every", dest="save_every", type=int, default=500)
    train(ap.parse_args())
