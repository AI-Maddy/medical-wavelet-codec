"""Fourier-Bessel series expansion (FBSE) analysis used to adapt the decomposition.

Implements the two building blocks of the Fourier-Bessel domain adaptive wavelet
transform (FBDAWT, Tripathy et al., IEEE Sensors Letters 2023) and uses them to
pick how many wavelet levels each image gets:

  (2) FBSE coefficients of a length-N signal x[n] (zero-order Bessel basis)
        c_m = 2 / (N^2 J1(b_m)^2) * sum_n n x[n] J0(b_m n / N),   m = 1..N
      where b_m is the m-th positive root of J0.  Order m corresponds to the
      normalised frequency f_m ~ b_m / (2 pi N) cycles/sample (0 .. 0.5).
      FBSE energy spectrum: E_m = c_m^2 N^2 J1(b_m)^2 / 2.

  (1) Empirical-wavelet scaling function between boundaries, with a cosine
      transition of half-width gamma*w_n around each boundary w_n
      (Meyer / EWT form used by the FBDAWT):
        phi(w) = 1                                    |w| <= (1-gamma) w_n
               = cos(pi/2 * beta((|w| - (1-gamma) w_n) / (2 gamma w_n)))
                                                      inside the transition
               = 0                                    otherwise
      with beta(t) = t^4 (35 - 84 t + 70 t^2 - 20 t^3).

Boundary detection follows the EWT recipe on the averaged FBSE spectrum of image
rows and columns: smooth, find local maxima, keep the largest ones, and place a
boundary at the minimum between consecutive kept maxima.

For integer lossless coding the analysis filters themselves must be lifting
steps (see lifting.py), so here the FBSE spectrum *steers* the transform: the
first adaptive boundary f_1 (the edge of the dominant low-frequency mode) sets
the number of dyadic levels, L = clip(floor(log2(0.5 / f_1)), L_min, L_max), so
that the last low band still just contains that mode.
"""
import numpy as np
from scipy.special import j0, j1, jn_zeros


_ROOTS = {}


def _roots(n):
    if n not in _ROOTS:
        _ROOTS[n] = jn_zeros(0, n)
    return _ROOTS[n]


def fbse_coefficients(x):
    """FBSE coefficients c_m (m = 1..N) of 1-D signals along the last axis."""
    x = np.asarray(x, dtype=np.float64)
    N = x.shape[-1]
    b = _roots(N)
    n = np.arange(N)
    basis = j0(np.outer(b, n) / N)                         # (N, N)
    c = (2.0 / (N ** 2 * j1(b) ** 2)) * ((x * n) @ basis.T)
    return c


def fbse_spectrum(x):
    """FBSE energy spectrum and the matching normalised frequencies."""
    N = x.shape[-1]
    b = _roots(N)
    c = fbse_coefficients(x)
    E = c ** 2 * N ** 2 * j1(b) ** 2 / 2.0
    f = b / (2 * np.pi * N)
    return f, E


def ewt_scaling(w, wn, gamma):
    """Equation (1): EWT/FBDAWT scaling function, vectorised over w (0..0.5)."""
    w = np.abs(np.asarray(w, dtype=np.float64))
    lo, hi = (1 - gamma) * wn, (1 + gamma) * wn
    t = np.clip((w - lo) / (2 * gamma * wn), 0, 1)
    beta = t ** 4 * (35 - 84 * t + 70 * t ** 2 - 20 * t ** 3)
    return np.where(w <= lo, 1.0, np.where(w >= hi, 0.0, np.cos(np.pi / 2 * beta)))


def image_spectrum(img, n_lines=64, length=256, rng=None):
    """Average FBSE spectrum over randomly chosen rows and columns (mean removed)."""
    rng = rng or np.random.default_rng(0)
    img = np.asarray(img, dtype=np.float64)
    H, W = img.shape
    segs = []
    n = min(length, H, W)
    for axis, (A, B) in ((0, (H, W)), (1, (W, H))):
        for _ in range(n_lines):
            i = rng.integers(0, A)
            j = rng.integers(0, B - n + 1)
            seg = img[i, j:j + n] if axis == 0 else img[j:j + n, i]
            segs.append(seg - seg.mean())
    S = np.stack(segs)
    f, E = fbse_spectrum(S)
    return f, E.mean(0)


def boundaries(f, E, n_modes=4, smooth=5):
    """EWT-style boundaries on a spectrum (local maxima -> minima between them)."""
    k = np.ones(smooth) / smooth
    s = np.convolve(np.log1p(E), k, mode="same")
    peaks = [i for i in range(1, len(s) - 1) if s[i] >= s[i - 1] and s[i] > s[i + 1]]
    if not peaks:
        return np.array([0.25])
    peaks = sorted(sorted(peaks, key=lambda i: -s[i])[:n_modes])
    bds = []
    for a, b in zip(peaks[:-1], peaks[1:]):
        bds.append(f[a + int(np.argmin(s[a:b + 1]))])
    if not bds:  # single dominant mode: use where the spectrum halves (in log) after it
        p = peaks[0]
        tail = np.where(s[p:] < s[p] - np.log(2))[0]
        bds = [f[p + tail[0]] if len(tail) else 0.25]
    return np.array(bds)


def choose_levels(img, lmin=2, lmax=6):
    f, E = image_spectrum(img)
    f1 = max(float(boundaries(f, E)[0]), 1e-3)
    L = int(np.floor(np.log2(0.5 / f1)))
    # also never shrink the low band below 8 pixels
    cap = int(np.floor(np.log2(min(img.shape) / 8)))
    return int(np.clip(L, lmin, min(lmax, cap))), f1
