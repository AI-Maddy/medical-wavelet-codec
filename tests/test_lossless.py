"""Bit-exactness tests (run: python -m pytest -q tests)."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wavecomp import codec, dicomio, fbse, lifting  # noqa: E402
from wavecomp.model import EntropyModel  # noqa: E402


def phantom(shape, dtype, seed=0):
    rng = np.random.default_rng(seed)
    H, W = shape
    yy, xx = np.mgrid[:H, :W]
    info = np.iinfo(dtype)
    lo, hi = max(info.min, -3000), min(info.max, 12000)
    img = lo + (hi - lo) * (0.5 + 0.4 * np.sin(xx / 7.0) * np.cos(yy / 11.0))
    img[(yy - H / 2) ** 2 + (xx - W / 2) ** 2 < (min(H, W) / 4) ** 2] = lo
    img += rng.normal(0, 30, shape)
    return np.clip(np.round(img), info.min, info.max).astype(dtype)


SHAPES = [(64, 64), (37, 51), (130, 97), (9, 300)]
DTYPES = [np.uint8, np.uint16, np.int16]


@pytest.mark.parametrize("family", ["legall53", "adaptive"])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("levels", [1, 3])
def test_lifting_invertible(family, shape, levels):
    x = np.random.default_rng(1).integers(-40000, 40000, shape)
    assert np.array_equal(lifting.inverse(lifting.forward(x, levels, family)), x)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES[:3])
def test_static_codec(dtype, shape):
    x = phantom(shape, dtype)
    for fam in ["legall53", "adaptive"]:
        assert np.array_equal(codec.decode_static(codec.encode_static(x, family=fam)), x)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return EntropyModel().eval()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES[:3])
def test_neural_codec(model, dtype, shape):
    x = phantom(shape, dtype, seed=2)
    blob = codec.encode(x, model, "ct")
    assert np.array_equal(codec.decode(blob, model), x)


def test_constant_image(model):
    x = np.full((40, 40), 77, np.uint16)
    assert np.array_equal(codec.decode(codec.encode(x, model, "mri"), model), x)
    assert np.array_equal(codec.decode_static(codec.encode_static(x)), x)


def test_dicom_wrapper(tmp_path, model):
    x = phantom((64, 80), np.int16)
    blob = codec.encode(x, model, "ct")
    dicomio.write(tmp_path / "a.dcm", blob, x)
    b2, ds = dicomio.read(tmp_path / "a.dcm")
    assert ds.Rows == 64 and ds.Columns == 80
    assert np.array_equal(codec.decode(b2, model), x)


def test_decode_in_fresh_process(tmp_path, model):
    """The decoder recomputes every probability; it must match in another process."""
    x = phantom((96, 96), np.uint16, seed=5)
    torch.save(dict(cfg=model.cfg, state=model.state_dict()), tmp_path / "m.pt")
    (tmp_path / "x.bin").write_bytes(codec.encode(x, model, "xray"))
    np.save(tmp_path / "x.npy", x)
    code = (f"import sys,numpy as np; sys.path.insert(0,{str(ROOT)!r}); from wavecomp import codec;"
            f"m=codec.load_model({str(tmp_path / 'm.pt')!r});"
            f"y=codec.decode(open({str(tmp_path / 'x.bin')!r},'rb').read(), m);"
            f"assert np.array_equal(y, np.load({str(tmp_path / 'x.npy')!r}))")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_fbse_reconstructs_smooth_signal():
    from scipy.special import j0
    N = 128
    n = np.arange(N)
    x = np.cos(2 * np.pi * 0.05 * n) + 0.5 * np.sin(2 * np.pi * 0.11 * n)
    c = fbse.fbse_coefficients(x)
    rec = (c[:, None] * j0(np.outer(fbse._roots(N), n) / N)).sum(0)
    assert np.abs(rec[5:-5] - x[5:-5]).max() < 0.15
    L, f1 = fbse.choose_levels(phantom((256, 256), np.uint16))
    assert 2 <= L <= 6 and 0 < f1 < 0.5


def test_histogram_packing(model):
    """Sparse code values (e.g. LUT-mapped radiographs) are packed and restored exactly."""
    rng = np.random.default_rng(3)
    levels = np.sort(rng.choice(16384, 700, replace=False)).astype(np.uint16)
    x = levels[np.clip(np.round(phantom((70, 90), np.uint16) / 20), 0, 699).astype(int)]
    packed, table = codec.pack(x)
    assert table is not None and packed.max() < 700
    assert np.array_equal(codec.decode(codec.encode(x, model, "xray"), model), x)
    assert np.array_equal(codec.decode_static(codec.encode_static(x)), x)


# ---- rANS and DMOL --------------------------------------------------------
from wavecomp import rans  # noqa: E402


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_rans_roundtrip(seed):
    rng = np.random.default_rng(seed)
    enc = rans.RansEncoder(rans.lanes_for(30000))
    calls = []
    for c in range(6):
        n = int(rng.integers(1, 9000))
        lo, hi = int(rng.integers(-5000, 0)), int(rng.integers(1, 70000))
        K = 5
        mu = rng.normal(0, 40, (n, K))
        sc = np.exp(rng.normal(1, 1.5, (n, K)))
        x = np.clip(np.round(mu[:, 0] + rng.laplace(0, 8, n)), lo, hi).astype(np.int64)
        x[: min(3, n)] = [lo, hi, lo][: min(3, n)]            # alphabet edges
        F = (rans.logistic_mixture_cdf(rng.normal(size=(n, K)), mu, sc) if c % 2
             else rans.laplace_cdf(mu[:, 0], sc[:, 0]))
        enc.add(x, F, lo, hi)
        calls.append((x, F, lo, hi))
    dec = rans.RansDecoder(enc.finish())
    for x, F, lo, hi in calls:
        assert np.array_equal(dec.decode(len(x), F, lo, hi), x)
    assert dec.check_final()


@pytest.fixture(scope="module")
def dmol_model():
    torch.manual_seed(0)
    return EntropyModel(mixture=5).eval()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES[:3])
def test_dmol_rans_codec(dmol_model, dtype, shape):
    x = phantom(shape, dtype, seed=3)
    blob = codec.encode(x, dmol_model, "mri")
    assert blob[:4] == codec.MAGIC_RANS
    assert np.array_equal(codec.decode(blob, dmol_model), x)


def test_laplace_rans_codec(model):
    x = phantom((130, 97), np.uint16, seed=4)
    blob = codec.encode(x, model, "ct", entropy_coder="rans")
    assert np.array_equal(codec.decode(blob, model), x)


def test_dmol_loss_matches_rans_size(dmol_model):
    """The training loss (ideal code length) and the real rANS stream agree to within the
    parameter quantisation and stream overhead."""
    x = phantom((128, 128), np.uint16, seed=5)
    work, _ = codec.pack(x)
    d = lifting.forward(work, 3, "adaptive", 4)
    tb = [torch.from_numpy(b.astype(np.int64))[None, None] for b in codec._bands_coarse_to_fine(d)]
    st = [(np.array([m]), np.array([s]), lo, hi) for (m, s, lo, hi) in codec.all_stats(d)]
    lows, cur = [], d.ll
    for l in range(2, 0, -1):
        cur = lifting.inverse_level(cur, d.bands[l], d.filters[l])
        lows.append(torch.from_numpy(cur)[None, None])
    coder = codec.TrainCoder()
    with torch.no_grad():
        codec.traverse(dmol_model, coder, torch.tensor([2]), st, codec._shapes(d.pad_shape, 3), tb,
                       lambda li, ll, three: lows[li])
    ideal = float(coder.bits) / x.size
    real = 8 * len(codec.encode(x, dmol_model, "mri", levels=3)) / x.size
    assert abs(real - ideal) < 0.15, (real, ideal)
