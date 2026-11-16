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
