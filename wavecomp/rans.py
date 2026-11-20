"""Interleaved range asymmetric numeral systems (rANS), vectorised with NumPy.

The coder works on any discrete distribution given as an integer cumulative
frequency function C(s) over an alphabet lo..hi, with C(lo) = 0, C(hi+1) = M and
C(s+1) - C(s) >= 1 for every symbol (so every value stays codable).

Design
  * 64-bit state per lane kept in [2**31, 2**63), 32-bit words, M = 2**31.
    With these constants at most one word moves per symbol and per lane.
  * N independent lanes (interleaved rANS). Inside one batch of n symbols
    (one call of ``RansEncoder.add`` / ``RansDecoder.decode``), symbol j goes to
    lane j % N, so all lanes advance together and NumPy does the work.
  * rANS is last-in first-out: the encoder only records (start, freq) per
    symbol and runs the actual coding backwards in ``finish``; the decoder then
    reads forward in the original order.

Stream: N (u16), per-lane word counts (u32 x N), final states (u64 x N), then the
words of lane 0, lane 1, ... in the order the decoder consumes them.
"""
import struct

import numpy as np

PREC = 31
M = 1 << PREC
L_LOW = np.uint64(1) << np.uint64(31)
MASK32 = np.uint64(0xFFFFFFFF)
S32 = np.uint64(32)
SPREC = np.uint64(PREC)
MMASK = np.uint64(M - 1)


def lanes_for(n_symbols):
    """About 4,096 symbols per lane, between 4 and 1,024 lanes (8 bytes of state each)."""
    return int(np.clip(n_symbols // 4096, 4, 1024))


# ---------------------------------------------------------------------------
# integer cumulative frequencies from a continuous CDF
# ---------------------------------------------------------------------------
def cum_freq(F, s, lo, hi):
    """C(s) for integer s (array), given F(y) = continuous CDF evaluated at y = s - 0.5.

    C(lo) = 0, C(hi + 1) = M, and in between floor(F(s - .5) * (M - R)) + (s - lo),
    which reserves one unit per symbol so no symbol has zero frequency.
    """
    R = hi - lo + 1
    s = np.asarray(s, dtype=np.int64)
    inner = np.floor(np.clip(F(s.astype(np.float64) - 0.5), 0.0, 1.0) * float(M - R)).astype(np.int64)
    c = inner + (s - lo)
    c = np.where(s <= lo, 0, c)
    c = np.where(s > hi, M, c)
    return c


def logistic_mixture_cdf(logits, mu, scale):
    """Return F(y, idx) for a K-component logistic mixture; params have shape (n, K)."""
    w = np.exp(logits - logits.max(1, keepdims=True))
    w = w / w.sum(1, keepdims=True)

    def F(y, idx):
        z = (y[:, None] - mu[idx]) / scale[idx]
        return (w[idx] / (1.0 + np.exp(-np.clip(z, -60, 60)))).sum(1)
    return F


def laplace_cdf(mu, b):
    def F(y, idx):
        z = (y - mu[idx]) / b[idx]
        return np.where(z < 0, 0.5 * np.exp(np.minimum(z, 0)), 1.0 - 0.5 * np.exp(-np.maximum(z, 0)))
    return F


# ---------------------------------------------------------------------------
# encoder
# ---------------------------------------------------------------------------
class RansEncoder:
    def __init__(self, n_lanes):
        self.N = int(n_lanes)
        self.batches = []        # (start, freq) int64 arrays in decode order

    def add(self, x, Fidx, lo, hi):
        """x: int array of symbols; Fidx(y, idx) the continuous CDF for symbols idx."""
        x = np.asarray(x, dtype=np.int64).ravel()
        if not len(x):
            return
        if x.min() < lo or x.max() > hi:
            raise ValueError("symbol outside the alphabet")
        idx = np.arange(len(x))
        F = lambda y: Fidx(y, idx)  # noqa: E731
        start = cum_freq(F, x, lo, hi)
        freq = cum_freq(F, x + 1, lo, hi) - start
        if freq.min() < 1:
            raise ValueError("zero-frequency symbol")
        self.batches.append((start, freq))

    def finish(self):
        N = self.N
        x = np.full(N, L_LOW, dtype=np.uint64)
        out_lane, out_word = [], []
        for start, freq in reversed(self.batches):
            n = len(start)
            steps = -(-n // N)
            for t in range(steps - 1, -1, -1):
                a, b = t * N, min(n, (t + 1) * N)
                lanes = np.arange(b - a)
                st = start[a:b].astype(np.uint64)
                fr = freq[a:b].astype(np.uint64)
                xs = x[lanes]
                xmax = fr << S32
                over = xs >= xmax
                if over.any():
                    out_lane.append(lanes[over])
                    out_word.append((xs[over] & MASK32).astype(np.uint32))
                    xs = np.where(over, xs >> S32, xs)
                x[lanes] = ((xs // fr) << SPREC) + (xs % fr) + st
        lane = np.concatenate(out_lane) if out_lane else np.zeros(0, np.int64)
        word = np.concatenate(out_word) if out_word else np.zeros(0, np.uint32)
        # words were emitted in reverse decode order: reverse, then group by lane (stable)
        lane, word = lane[::-1], word[::-1]
        order = np.argsort(lane, kind="stable")
        counts = np.bincount(lane, minlength=N).astype("<u4")
        return (struct.pack("<H", N) + counts.tobytes() + x.astype("<u8").tobytes()
                + word[order].astype("<u4").tobytes())


# ---------------------------------------------------------------------------
# decoder
# ---------------------------------------------------------------------------
class RansDecoder:
    def __init__(self, buf):
        N, = struct.unpack_from("<H", buf, 0)
        self.N = N
        o = 2
        counts = np.frombuffer(buf, dtype="<u4", count=N, offset=o).astype(np.int64)
        o += 4 * N
        self.x = np.frombuffer(buf, dtype="<u8", count=N, offset=o).astype(np.uint64)
        o += 8 * N
        self.words = np.frombuffer(buf, dtype="<u4", count=int(counts.sum()), offset=o).astype(np.uint64)
        self.ptr = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
        self.end = self.ptr + counts

    def decode(self, n, Fidx, lo, hi):
        N = self.N
        out = np.empty(n, dtype=np.int64)
        steps = -(-n // N)
        for t in range(steps):
            a, b = t * N, min(n, (t + 1) * N)
            lanes = np.arange(b - a)
            idx = np.arange(a, b)
            F = lambda y: Fidx(y, idx)  # noqa: E731
            xs = self.x[lanes]
            slot = (xs & MMASK).astype(np.int64)
            # binary search: largest s in [lo, hi] with C(s) <= slot
            s_lo = np.full(b - a, lo, dtype=np.int64)
            s_hi = np.full(b - a, hi + 1, dtype=np.int64)
            while True:
                act = s_hi - s_lo > 1
                if not act.any():
                    break
                mid = (s_lo + s_hi) // 2
                c = cum_freq(F, mid, lo, hi)
                go = c <= slot
                s_lo = np.where(act & go, mid, s_lo)
                s_hi = np.where(act & ~go, mid, s_hi)
            s = s_lo
            start = cum_freq(F, s, lo, hi)
            freq = cum_freq(F, s + 1, lo, hi) - start
            xs = freq.astype(np.uint64) * (xs >> SPREC) + (slot - start).astype(np.uint64)
            under = xs < L_LOW
            if under.any():
                ul = lanes[under]
                p = self.ptr[ul]
                if (p >= self.end[ul]).any():
                    raise ValueError("rANS stream exhausted (corrupt stream)")
                xs[under] = (xs[under] << S32) | self.words[p]
                self.ptr[ul] = p + 1
            self.x[lanes] = xs
            out[a:b] = s
        return out

    def check_final(self):
        """After the last symbol every lane must be back at its initial state."""
        return bool((self.x == L_LOW).all() and (self.ptr == self.end).all())
