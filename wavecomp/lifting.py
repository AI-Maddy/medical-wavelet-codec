"""Integer-to-integer 2-D lifting wavelet transforms (exactly invertible).

Every lifting step has the form

    d[n] = o[n] - ((sum_j P_j * e[n + j] + 2**(S-1)) >> S)          (predict)
    s[n] = e[n] + ((sum_j U_j * d[n + j] + 2**(S-1)) >> S)          (update)

with integer taps P_j, U_j on a common denominator 2**S (S = 6, i.e. 1/64 steps).
Because the right-hand sides use only integer arithmetic on values the decoder
already has, the inverse simply undoes the steps in reverse order and is
bit-exact on every platform, whatever the tap values are.  This is what lets the
predict filters be *adapted* to each image (fitted by iteratively re-weighted
least squares on the image itself) without losing perfect reconstruction.

Two filter families are provided:
  * "legall53": the reversible LeGall 5/3 wavelet of JPEG 2000 (fixed taps).
  * "adaptive": per-image, per-level, per-direction predict filters with 2*T taps,
    fitted to minimise the L1 norm of the detail coefficients (IRLS), quantised to
    1/64, and sent in the stream header.  The update step stays LeGall's
    [1/4, 1/4] so the low band keeps its anti-aliasing behaviour across levels.
"""
from dataclasses import dataclass, field

import numpy as np

SHIFT = 6
ONE = 1 << SHIFT
HALF = 1 << (SHIFT - 1)

LEGALL_P = np.array([ONE // 2, ONE // 2], dtype=np.int64)        # offsets 0, +1
LEGALL_U = np.array([ONE // 4, ONE // 4], dtype=np.int64)        # offsets -1, 0


def _reflect(idx, m):
    """Whole-sample symmetric index reflection into [0, m)."""
    if m == 1:
        return np.zeros_like(idx)
    period = 2 * (m - 1)
    idx = np.abs(idx) % period
    return np.where(idx >= m, period - idx, idx)


def _taps(arr, offsets, axis):
    """Stack arr shifted by each offset along axis (reflected at the borders)."""
    m = arr.shape[axis]
    base = np.arange(m)
    return [np.take(arr, _reflect(base + k, m), axis=axis) for k in offsets]


def predict_offsets(ntaps):
    t = ntaps // 2
    return list(range(-(t - 1), t + 1))          # e.g. 4 taps -> -1, 0, 1, 2


UPDATE_OFFSETS = [-1, 0]


def _apply_filter(arr, taps, offsets, axis):
    acc = np.zeros(arr.shape, dtype=np.int64)
    for c, sh in zip(taps, _taps(arr, offsets, axis)):
        if c:
            acc += int(c) * sh
    return (acc + HALF) >> SHIFT


def fwd_1d(x, p, axis):
    """One forward lifting step along axis (length must be even)."""
    x = x.astype(np.int64, copy=False)
    e = np.take(x, np.arange(0, x.shape[axis], 2), axis=axis)
    o = np.take(x, np.arange(1, x.shape[axis], 2), axis=axis)
    d = o - _apply_filter(e, p, predict_offsets(len(p)), axis)
    s = e + _apply_filter(d, LEGALL_U, UPDATE_OFFSETS, axis)
    return s, d


def inv_1d(s, d, p, axis):
    e = s - _apply_filter(d, LEGALL_U, UPDATE_OFFSETS, axis)
    o = d + _apply_filter(e, p, predict_offsets(len(p)), axis)
    shape = list(s.shape)
    shape[axis] *= 2
    x = np.empty(shape, dtype=np.int64)
    sl_e = [slice(None)] * x.ndim
    sl_o = [slice(None)] * x.ndim
    sl_e[axis] = slice(0, None, 2)
    sl_o[axis] = slice(1, None, 2)
    x[tuple(sl_e)] = e
    x[tuple(sl_o)] = o
    return x


def fit_predict(x, axis, ntaps, iters=4, max_rows=512, rng=None):
    """Fit integer predict taps (sum = 1) minimising sum |d| by IRLS."""
    x = x.astype(np.float64)
    if axis == 0:
        x = x.T
    if x.shape[0] > max_rows:  # subsample rows for speed; taps barely change
        rng = rng or np.random.default_rng(0)
        x = x[np.sort(rng.choice(x.shape[0], max_rows, replace=False))]
    e = x[:, 0::2]
    o = x[:, 1::2]
    offs = predict_offsets(ntaps)
    A = np.stack([sh.ravel() for sh in _taps(e, offs, 1)], 1)
    b = o.ravel()
    # constraint sum(p) = 1 (DC gain 1): substitute p_last = 1 - sum(others)
    A2 = A[:, :-1] - A[:, -1:]
    b2 = b - A[:, -1]
    w = np.ones_like(b2)
    p = np.full(ntaps - 1, 1.0 / ntaps)
    for _ in range(iters):
        sw = np.sqrt(w)
        p, *_ = np.linalg.lstsq(A2 * sw[:, None], b2 * sw, rcond=None)
        r = np.abs(b2 - A2 @ p)
        w = 1.0 / np.maximum(r, 1.0)
    full = np.append(p, 1.0 - p.sum())
    q = np.round(full * ONE).astype(np.int64)
    q[-1] += ONE - q.sum()           # exact DC gain after rounding
    return np.clip(q, -4 * ONE, 4 * ONE)


@dataclass
class Decomposition:
    """Integer wavelet pyramid. bands[l] = (HL, LH, HH) at level l+1 (finest first)."""
    ll: np.ndarray
    bands: list
    filters: list = field(default_factory=list)   # per level: (p_rows, p_cols_low, p_cols_high)
    shape: tuple = ()
    pad_shape: tuple = ()

    @property
    def levels(self):
        return len(self.bands)


def pad_to(x, levels):
    m = 1 << levels
    H, W = x.shape
    ph, pw = (-H) % m, (-W) % m
    if ph or pw:
        x = np.pad(x, ((0, ph), (0, pw)), mode="edge")
    return x


def choose_predict(x, axis, ntaps):
    """Closed-loop choice: fitted taps for 2..ntaps, or LeGall, whichever gives the
    smallest sum |d| on this signal.  Never worse than the fixed 5/3 wavelet."""
    cands = [LEGALL_P] + [fit_predict(x, axis, t) for t in range(2, ntaps + 1, 2)]
    costs = []
    for p in cands:
        e = np.take(x, np.arange(0, x.shape[axis], 2), axis=axis)
        o = np.take(x, np.arange(1, x.shape[axis], 2), axis=axis)
        costs.append(np.abs(o - _apply_filter(e, p, predict_offsets(len(p)), axis)).sum())
    return cands[int(np.argmin(costs))]


def forward(x, levels, family="adaptive", ntaps=4):
    """x: 2-D integer image. Returns Decomposition (pads to a multiple of 2**levels)."""
    shape = x.shape
    cur = pad_to(np.asarray(x, dtype=np.int64), levels)
    pad_shape = cur.shape
    bands, filters = [], []
    for _ in range(levels):
        if family == "legall53":
            pr = pcl = pch = LEGALL_P
        else:
            pr = choose_predict(cur, 1, ntaps)
        L, H = fwd_1d(cur, pr, axis=1)
        if family != "legall53":
            pcl = choose_predict(L, 0, ntaps)
            pch = choose_predict(H, 0, ntaps)
        LL, LH = fwd_1d(L, pcl, axis=0)
        HL, HH = fwd_1d(H, pch, axis=0)
        bands.append((HL, LH, HH))
        filters.append((pr, pcl, pch))
        cur = LL
    return Decomposition(cur, bands, filters, shape, pad_shape)


def inverse_level(ll, band, filt):
    HL, LH, HH = band
    pr, pcl, pch = filt
    L = inv_1d(ll, LH, pcl, axis=0)
    H = inv_1d(HL, HH, pch, axis=0)
    return inv_1d(L, H, pr, axis=1)


def inverse(dec):
    cur = dec.ll
    for band, filt in zip(reversed(dec.bands), reversed(dec.filters)):
        cur = inverse_level(cur, band, filt)
    H, W = dec.shape
    return cur[:H, :W]
