"""Lossless codec: adaptive integer wavelet + neural entropy model + range coding.

Public API
    encode(img, model, modality, ...) -> bytes
    decode(blob, model) -> np.ndarray       (bit-exact)
    encode_static(img, ...) / decode_static(blob)
        the "conventional" reference: same transform, but each subband is coded
        with one static discretised Laplace whose parameters are in the header.

The same traversal routine (``traverse``) drives training, encoding and
decoding, so the probabilities the decoder computes are by construction the
ones the encoder used.
"""
import io
import struct
import zlib

import constriction
import numpy as np
import torch

from . import fbse, lifting, rans
from .model import MODALITIES, EntropyModel, upsample

MAGIC = b"WVC1"            # range coder (constriction)
MAGIC_RANS = b"WVR1"       # interleaved rANS (wavecomp.rans)
FAMILIES = ["legall53", "adaptive"]
DTYPES = ["uint8", "int8", "uint16", "int16", "uint32", "int32"]
LL_DEPTH = 32          # recursion depth for the low band (stops at 8 px anyway)
DETAIL_DEPTH = 2       # polyphase recursion depth for detail subbands
MU_Q = 16.0            # mean quantised to 1/16
LOGB_Q = 32.0          # log2(scale) quantised to 1/32
PI_Q = 16.0            # DMOL mixture logits quantised to 1/16
_Q = constriction.stream.queue
_M = constriction.stream.model


# ---------------------------------------------------------------------------
# subband statistics carried in the header
# ---------------------------------------------------------------------------
def band_stats(x, is_ll):
    m = int(np.round(x.mean())) if is_ll else 0
    s = float(np.float16(max(np.abs(x - m).mean(), 0.5)))
    return m, s, int(x.min()), int(x.max())


def all_stats(dec):
    st = [band_stats(dec.ll, True)]
    for l in range(dec.levels - 1, -1, -1):          # coarse -> fine, HL LH HH
        for b in dec.bands[l]:
            st.append(band_stats(b, False))
    return st


# ---------------------------------------------------------------------------
# "coders": the three ways of consuming the model's probabilities
# ---------------------------------------------------------------------------
def laplace_nll_bits(x, mu, b):
    """-log2 P(x) for a Laplace(mu, b) discretised to integer bins."""
    z = x - mu
    a = torch.abs(z)
    # P = F(x+.5) - F(x-.5)
    p_in = 1 - torch.exp(-0.5 / b) * torch.cosh(torch.clamp(z / b, -30, 30))
    log_out = -(a - 0.5) / b + torch.log(0.5 * (1 - torch.exp(-1 / b)) + 1e-30)
    logp = torch.where(a < 0.5, torch.log(torch.clamp(p_in, min=1e-12)), log_out)
    return -torch.clamp(logp, min=np.log(1e-12)) / np.log(2)


def quantise_params(mu, b, lo, hi):
    mu = np.clip(np.round(mu.astype(np.float64) * MU_Q) / MU_Q, lo - 1, hi + 1)
    lb = np.clip(np.round(np.log2(np.maximum(b.astype(np.float64), 1e-6)) * LOGB_Q) / LOGB_Q, -4, 24)
    return mu, np.exp2(lb)


def dmol_nll_bits(x, logits, mu, scale, lo, hi):
    """-log2 P(x) for a discretised mixture of logistics on the integers lo..hi.

    The first and last bins absorb the tails, as in PixelCNN++ (Salimans et al. 2017).
    x: (B,1,H,W); logits, mu, scale: (B,K,H,W); lo, hi: per-sample arrays.
    """
    F_ = torch.nn.functional
    lo = _t(lo)
    hi = _t(hi)
    inv = 1.0 / scale
    c = x - mu
    up = inv * (c + 0.5)
    dn = inv * (c - 0.5)
    log_cdf_up = up - F_.softplus(up)                 # log sigmoid(up)
    log_1m_cdf_dn = -F_.softplus(dn)                  # log (1 - sigmoid(dn))
    diff = torch.sigmoid(up) - torch.sigmoid(dn)
    mid = inv * c
    log_pdf_mid = mid - torch.log(scale) - 2.0 * F_.softplus(mid)
    log_mid = torch.where(diff > 1e-5, torch.log(torch.clamp(diff, min=1e-12)), log_pdf_mid)
    logp = torch.where(x <= lo, log_cdf_up, torch.where(x >= hi, log_1m_cdf_dn, log_mid))
    logp = torch.logsumexp(logp + torch.log_softmax(logits, 1), 1, keepdim=True)
    return -torch.clamp(logp, min=-31 * np.log(2)) / np.log(2)


def quantise_mixture(logits, mu, scale, lo, hi):
    """Quantise DMOL parameters (shape (n, K)) so encoder and decoder build the same CDF."""
    lg = np.clip(np.round(logits.astype(np.float64) * PI_Q) / PI_Q, -16, 16)
    mu = np.clip(np.round(mu.astype(np.float64) * MU_Q) / MU_Q, lo - 1, hi + 1)
    ls = np.clip(np.round(np.log2(np.maximum(scale.astype(np.float64), 1e-6)) * LOGB_Q) / LOGB_Q, -4, 24)
    return lg, mu, np.exp2(ls)


def _flat_k(t):
    """(1,K,h,w) tensor -> (h*w, K) float64 array."""
    return t.detach().numpy()[0].reshape(t.shape[1], -1).T.astype(np.float64)


def rans_cdf(P, lo, hi):
    """Continuous CDF F(y, idx) for the quantised distribution P (Laplace or DMOL)."""
    if len(P) == 2:
        mu_q, b_q = quantise_params(P[0].detach().numpy().ravel(), P[1].detach().numpy().ravel(), lo, hi)
        return rans.laplace_cdf(mu_q, b_q)
    lg, mu, sc = quantise_mixture(_flat_k(P[0]), _flat_k(P[1]), _flat_k(P[2]), lo, hi)
    return rans.logistic_mixture_cdf(lg, mu, sc)


class TrainCoder:
    def __init__(self):
        self.bits = 0.0

    def code(self, x, P, lo, hi):
        if len(P) == 2:
            self.bits = self.bits + laplace_nll_bits(x.float(), *P).sum()
        else:
            self.bits = self.bits + dmol_nll_bits(x.float(), *P, lo, hi).sum()
        return x


class EncCoder:
    def __init__(self):
        self.enc = _Q.RangeEncoder()

    def code(self, x, P, lo, hi):
        if int(lo) == int(hi):
            return x
        if len(P) != 2:
            raise ValueError("the range coder supports the Laplace model only; use entropy_coder='rans'")
        mu, b = P
        model = _M.QuantizedLaplace(int(lo), int(hi))
        mu_q, b_q = quantise_params(mu.detach().numpy().ravel(), b.detach().numpy().ravel(), lo, hi)
        self.enc.encode(x.numpy().ravel().astype(np.int32), model, mu_q, b_q)
        return x


class DecCoder:
    def __init__(self, words):
        self.dec = _Q.RangeDecoder(words)

    def code(self, x, P, lo, hi):
        mu, b = P
        if int(lo) == int(hi):
            return torch.full(mu.shape, int(lo), dtype=torch.int64)
        model = _M.QuantizedLaplace(int(lo), int(hi))
        mu_q, b_q = quantise_params(mu.detach().numpy().ravel(), b.detach().numpy().ravel(), lo, hi)
        v = self.dec.decode(model, mu_q, b_q)
        return torch.from_numpy(v.astype(np.int64)).reshape(mu.shape)


class RansEncCoder:
    def __init__(self, n_lanes):
        self.enc = rans.RansEncoder(n_lanes)

    def code(self, x, P, lo, hi):
        lo, hi = int(lo), int(hi)
        if lo == hi:
            return x
        self.enc.add(x.numpy().ravel(), rans_cdf(P, lo, hi), lo, hi)
        return x


class RansDecCoder:
    def __init__(self, buf):
        self.dec = rans.RansDecoder(buf)

    def code(self, x, P, lo, hi):
        lo, hi = int(lo), int(hi)
        shape = P[0].shape[:1] + (1,) + P[0].shape[2:]
        if lo == hi:
            return torch.full(shape, lo, dtype=torch.int64)
        n = int(np.prod(shape))
        v = self.dec.decode(n, rans_cdf(P, lo, hi), lo, hi)
        return torch.from_numpy(v).reshape(shape)


# ---------------------------------------------------------------------------
# traversal shared by training / encoding / decoding
# ---------------------------------------------------------------------------
def code_plane(model, coder, feat, xtrue, depth, m, s, lo, hi):
    """Code one plane (B,1,H,W) in recursive 2x2 polyphase order."""
    B, _, H, W = feat.shape

    def nrm(v):
        return (v.float() - m) / s

    if depth == 0 or min(H, W) < 8:
        P = model.params(feat, torch.zeros(B, 1, H, W), torch.zeros(B, 1, H, W), 0)
        return coder.code(xtrue, _scale(P, m, s), lo, hi)
    x00 = code_plane(model, coder, feat[..., ::2, ::2],
                     None if xtrue is None else xtrue[..., ::2, ::2], depth - 1, m, s, lo, hi)
    out = torch.zeros(B, 1, H, W, dtype=torch.int64)
    known = torch.zeros(B, 1, H, W)
    mask = torch.zeros(B, 1, H, W)
    out[..., ::2, ::2] = x00
    known[..., ::2, ::2] = nrm(x00)
    mask[..., ::2, ::2] = 1
    for k, (dy, dx) in enumerate([(1, 1), (0, 1), (1, 0)]):
        if H <= dy or W <= dx:
            continue
        P = model.params(feat, known, mask, k + 1)
        sl = (Ellipsis, slice(dy, None, 2), slice(dx, None, 2))
        xs = coder.code(None if xtrue is None else xtrue[sl], _scale(tuple(p[sl] for p in P), m, s), lo, hi)
        out[sl] = xs
        known[sl] = nrm(xs)
        mask[sl] = 1
    return out


def _scale(P, m, s):
    """Model outputs (normalised units) -> distribution parameters in coefficient units."""
    if len(P) == 2:
        mu, logb = P
        return mu * s + m, s * torch.exp(logb)
    logits, mu, logs = P
    return logits, mu * s + m, s * torch.exp(logs)


def _t(v):
    return torch.as_tensor(v, dtype=torch.float32).view(-1, 1, 1, 1)


def traverse(model, coder, mod, stats, shapes, true_bands=None, low_fn=None):
    """Code the pyramid coarse -> fine.

    stats   list over subbands (LL, then HL LH HH per level, coarse->fine) of
            per-sample tuples (m, s, lo, hi) as arrays of shape (B,)
    shapes  list of (H, W) per level, coarse -> fine (LL_L shape first)
    true_bands  same order as stats, tensors (B,1,H,W) (None when decoding)
    low_fn(level_idx, ll, bands3) -> next low band (B,1,2H,2W), integer tensor
    Returns the list of coded subbands (true or decoded).
    """
    B = mod.shape[0]
    L = len(shapes)
    coded = []
    it = iter(range(len(stats)))

    def nxt():
        i = next(it)
        m, s, lo, hi = stats[i]
        x = None if true_bands is None else true_bands[i]
        return x, _t(m), _t(s), lo, hi

    # --- LL_L ---------------------------------------------------------------
    H, W = shapes[0]
    h, c = model.zero_state(B, H, W)
    z1, z3 = torch.zeros(B, 1, H, W), torch.zeros(B, 3, H, W)
    feat = model.context(z1, z3, z3, h, 0, mod)
    x, m, s, lo, hi = nxt()
    ll = code_plane(model, coder, feat, x, LL_DEPTH, m, s, lo, hi)
    coded.append(ll)
    h, c = model.step(feat, (ll.float() - m) / s, h, c)
    low = ll
    parents = None
    for li in range(L):
        H, W = shapes[li]
        if li > 0:
            h, c = upsample(h, (H, W)), upsample(c, (H, W))
        lowf = low.float()
        mean = lowf.mean((1, 2, 3), keepdim=True)
        std = lowf.std((1, 2, 3), keepdim=True) + 1.0
        lown = (lowf - mean) / std
        par = torch.zeros(B, 3, H, W) if parents is None else upsample(parents, (H, W))
        sibs = torch.zeros(B, 3, H, W)
        three = []
        for bi in range(3):
            feat = model.context(lown, sibs, par, h, bi + 1, mod)
            x, m, s, lo, hi = nxt()
            xb = code_plane(model, coder, feat, x, DETAIL_DEPTH, m, s, lo, hi)
            xn = (xb.float() - m) / s
            sibs = sibs.clone()
            sibs[:, bi:bi + 1] = xn
            h, c = model.step(feat, xn, h, c)
            three.append(xb)
            coded.append(xb)
        parents = sibs
        if li + 1 < L:
            low = low_fn(li, low, three)
    return coded


# ---------------------------------------------------------------------------
# histogram packing (reversible intensity remapping)
# ---------------------------------------------------------------------------
PACK_RATIO = 0.5


def pack(img):
    """Many radiographs use only a sparse subset of their code values (e.g. after a
    look-up table).  Map the used values onto 0..K-1 when K < PACK_RATIO * span;
    the sorted value table travels in the header (zlib-compressed deltas)."""
    img = np.asarray(img)
    u = np.unique(img)
    span = int(u[-1]) - int(u[0]) + 1
    if len(u) >= PACK_RATIO * span or len(u) < 2:
        return img, None
    packed = np.searchsorted(u, img).astype(np.int64)
    deltas = np.diff(u.astype(np.int64), prepend=0)
    table = zlib.compress(deltas.astype("<i4").tobytes(), 9)
    return packed, table


def unpack(rec, table):
    if table is None:
        return rec
    u = np.cumsum(np.frombuffer(zlib.decompress(table), dtype="<i4").astype(np.int64))
    return u[rec]


# ---------------------------------------------------------------------------
# header
# ---------------------------------------------------------------------------
def _write_header(f, img, L, family, mod, dec, stats, crc, table=None, magic=MAGIC):
    H, W = img.shape
    f.write(magic)
    f.write(struct.pack("<IIBBBB", H, W, DTYPES.index(str(img.dtype)), L, FAMILIES.index(family),
                        MODALITIES.index(mod)))
    f.write(struct.pack("<I", crc))
    f.write(struct.pack("<I", 0 if table is None else len(table)))
    if table is not None:
        f.write(table)
    if family == "adaptive":
        for filt in reversed(dec.filters):          # coarse -> fine
            for p in filt:
                f.write(struct.pack("<B", len(p)))
                f.write(np.asarray(p, dtype="<i2").tobytes())
    for (m, s, lo, hi) in stats:
        f.write(struct.pack("<ieii", m, s, lo, hi))


def _read_header(f):
    magic = f.read(4)
    assert magic in (MAGIC, MAGIC_RANS), "not a WVC1/WVR1 stream"
    H, W, dt, L, fam, mod = struct.unpack("<IIBBBB", f.read(12))
    crc, = struct.unpack("<I", f.read(4))
    ntab, = struct.unpack("<I", f.read(4))
    table = f.read(ntab) if ntab else None
    family = FAMILIES[fam]
    filters = []
    if family == "adaptive":
        for _ in range(L):
            filt = []
            for _ in range(3):
                n, = struct.unpack("<B", f.read(1))
                filt.append(np.frombuffer(f.read(2 * n), dtype="<i2").astype(np.int64))
            filters.append(tuple(filt))
        filters = filters[::-1]                       # back to fine -> coarse
    else:
        filters = [(lifting.LEGALL_P,) * 3] * L
    stats = [struct.unpack("<ieii", f.read(14)) for _ in range(1 + 3 * L)]
    return dict(magic=magic, H=H, W=W, dtype=DTYPES[dt], L=L, family=family, mod=MODALITIES[mod], crc=crc, table=table,
                filters=filters, stats=stats)


def _shapes(pad_shape, L):
    H, W = pad_shape
    return [(H >> l, W >> l) for l in range(L, 0, -1)]


def _prepare(img, levels, family, ntaps):
    if levels is None:
        levels, _ = fbse.choose_levels(img)
    dec = lifting.forward(img, levels, family, ntaps)
    return dec, levels


def _bands_coarse_to_fine(dec):
    out = [dec.ll]
    for l in range(dec.levels - 1, -1, -1):
        out.extend(dec.bands[l])
    return out


# ---------------------------------------------------------------------------
# neural codec
# ---------------------------------------------------------------------------
@torch.no_grad()
def encode(img, model, modality, levels=None, family="adaptive", ntaps=4, entropy_coder=None):
    """entropy_coder: "range" (constriction range coder, Laplace model only) or "rans"
    (interleaved rANS, any model). Default: "rans" for a DMOL model, else "range"."""
    if entropy_coder is None:
        entropy_coder = "rans" if getattr(model, "mixture", 0) else "range"
    img = np.asarray(img)
    work, table = pack(img)
    dec, L = _prepare(work, levels, family, ntaps)
    stats = all_stats(dec)
    f = io.BytesIO()
    _write_header(f, img, L, family, modality, dec, stats, zlib.crc32(img.tobytes()), table,
                  MAGIC_RANS if entropy_coder == "rans" else MAGIC)
    coder = RansEncCoder(rans.lanes_for(img.size)) if entropy_coder == "rans" else EncCoder()
    tb = [torch.from_numpy(b.astype(np.int64))[None, None] for b in _bands_coarse_to_fine(dec)]
    st = [(np.array([m]), np.array([s]), lo, hi) for (m, s, lo, hi) in stats]
    mod = torch.tensor([MODALITIES.index(modality)])
    filters_c2f = dec.filters[::-1]

    def low_fn(li, ll, three):
        return torch.from_numpy(lifting.inverse_level(
            ll[0, 0].numpy(), tuple(t[0, 0].numpy() for t in three), filters_c2f[li]))[None, None]

    traverse(model.eval(), coder, mod, st, _shapes(dec.pad_shape, L), tb, low_fn)
    if entropy_coder == "rans":
        f.write(coder.enc.finish())
    else:
        f.write(coder.enc.get_compressed().astype("<u4").tobytes())
    return f.getvalue()


@torch.no_grad()
def decode(blob, model):
    f = io.BytesIO(blob)
    hd = _read_header(f)
    rest = f.read()
    L = hd["L"]
    m_ = 1 << L
    pad_shape = (hd["H"] + (-hd["H"]) % m_, hd["W"] + (-hd["W"]) % m_)
    if hd["magic"] == MAGIC_RANS:
        coder = RansDecCoder(rest)
    else:
        coder = DecCoder(np.frombuffer(rest[:len(rest) // 4 * 4], dtype="<u4").astype(np.uint32))
    st = [(np.array([m]), np.array([s]), lo, hi) for (m, s, lo, hi) in hd["stats"]]
    mod = torch.tensor([MODALITIES.index(hd["mod"])])
    filters_c2f = hd["filters"][::-1]

    def low_fn(li, ll, three):
        return torch.from_numpy(lifting.inverse_level(
            ll[0, 0].numpy(), tuple(t[0, 0].numpy() for t in three), filters_c2f[li]))[None, None]

    coded = traverse(model.eval(), coder, mod, st, _shapes(pad_shape, L), None, low_fn)
    if hd["magic"] == MAGIC_RANS and not coder.dec.check_final():
        raise ValueError("rANS stream not fully consumed: corrupt stream or model mismatch")
    ll = coded[0][0, 0].numpy()
    bands = []
    k = 1
    for _ in range(L):
        bands.append(tuple(coded[k + j][0, 0].numpy() for j in range(3)))
        k += 3
    bands = bands[::-1]                                 # fine -> coarse
    d = lifting.Decomposition(ll, bands, hd["filters"], (hd["H"], hd["W"]), pad_shape)
    out = unpack(lifting.inverse(d), hd["table"]).astype(hd["dtype"])
    if zlib.crc32(out.tobytes()) != hd["crc"]:
        raise ValueError("CRC mismatch: decoded image differs from the original")
    return out


# ---------------------------------------------------------------------------
# conventional reference: static per-subband Laplace (no context modelling)
# ---------------------------------------------------------------------------
def encode_static(img, levels=None, family="legall53", ntaps=4):
    img = np.asarray(img)
    work, table = pack(img)
    dec, L = _prepare(work, levels, family, ntaps)
    stats = all_stats(dec)
    f = io.BytesIO()
    _write_header(f, img, L, family, "xray", dec, stats, zlib.crc32(img.tobytes()), table)
    enc = _Q.RangeEncoder()
    for band, (m, s, lo, hi) in zip(_bands_coarse_to_fine(dec), stats):
        if lo == hi:
            continue
        enc.encode(band.ravel().astype(np.int32), _M.QuantizedLaplace(lo, hi, float(m), s))
    f.write(enc.get_compressed().astype("<u4").tobytes())
    return f.getvalue()


def decode_static(blob):
    f = io.BytesIO(blob)
    hd = _read_header(f)
    rest = f.read()
    words = np.frombuffer(rest[:len(rest) // 4 * 4], dtype="<u4").astype(np.uint32)
    L = hd["L"]
    m_ = 1 << L
    pad_shape = (hd["H"] + (-hd["H"]) % m_, hd["W"] + (-hd["W"]) % m_)
    shapes = _shapes(pad_shape, L)
    dec = _Q.RangeDecoder(words)
    sizes = [shapes[0]] + [s for s in shapes for _ in range(3)]
    vals = []
    for (m, s, lo, hi), shp in zip(hd["stats"], sizes):
        n = shp[0] * shp[1]
        if lo == hi:
            vals.append(np.full(shp, lo, dtype=np.int64))
            continue
        vals.append(dec.decode(_M.QuantizedLaplace(lo, hi, float(m), s), n).astype(np.int64).reshape(shp))
    bands = [tuple(vals[1 + 3 * i + j] for j in range(3)) for i in range(L)][::-1]
    d = lifting.Decomposition(vals[0], bands, hd["filters"], (hd["H"], hd["W"]), pad_shape)
    out = unpack(lifting.inverse(d), hd["table"]).astype(hd["dtype"])
    if zlib.crc32(out.tobytes()) != hd["crc"]:
        raise ValueError("CRC mismatch")
    return out


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = EntropyModel(**ck["cfg"])
    m.load_state_dict(ck["state"])
    return m.eval()
