# medical-wavelet-codec

Code for *"AI-Enhanced Lossless Compression of Medical Images Using Adaptive Wavelet
Transforms and Neural Entropy Coding"* (M. Vivekanandan).

This repository contains only the Python code that produces the reported results: the codec,
the training and evaluation scripts, the two trained models, and the measured outputs in
`results/`. Every rate is the size of a real, decodable bit stream, headers included. Every
test image was decoded again and checked bit-for-bit against the original.

## Which script produces which result

| Result | Script | Output |
|---|---|---|
| Compression per dataset and comparison with PNG, JPEG-LS, JPEG 2000 and JPEG XL (bpp, CR, encode/decode time) | `scripts/evaluate.py` then `scripts/summarize.py` | `results/per_image.csv`, `results/summary.md`, `results/summary.json` |
| Ablation of each component (configurations A–G below) | `scripts/evaluate.py` then `scripts/summarize.py` | same files, rows A–G |
| Ablation figure (bpp of A–E against the standard codecs) | `scripts/summarize.py` | `results/fig_ablation.png`, `.pdf` |
| Lossless reconstruction check (original, decoded, zero difference) | `scripts/make_figures.py` | `results/fig_lossless.png` |
| Fourier-Bessel spectrum, boundaries and chosen decomposition depth | `scripts/make_figures.py` | `results/fig_fbse.png` |
| Trained entropy models used by all of the above | `scripts/train.py` | `models/full.pt`, `models/plain.pt` |
| Public test data | `scripts/fetch_idc.py` | `data/` (not committed) |

Ablation configurations:

| | Transform | Depth | Entropy coding |
|---|---|---|---|
| A | LeGall 5/3 integer DWT | 5 levels | static Laplace per subband |
| B | adaptive lifting | 5 levels | static Laplace per subband |
| C | adaptive lifting | FBSE-chosen | static Laplace per subband |
| D | adaptive lifting | FBSE-chosen | neural context model, no ConvLSTM, no attention |
| E | adaptive lifting | FBSE-chosen | ConvLSTM neural entropy model with modality attention (proposed) |
| F | LeGall 5/3 | 5 levels | as E |
| G | adaptive lifting | 5 levels | as E |

## Measured results (held-out test patients)

Bits per pixel over all test pixels (lower is better), with CR = 16 / bpp. The full tables,
including timings, are in [`results/summary.md`](results/summary.md).

| Codec | Chest X-ray | Thoracic CT | Brain MRI |
|---|---:|---:|---:|
| PNG | 11.14 | 8.68 | 8.02 |
| JPEG 2000 lossless | 8.82 | 6.47 | 6.02 |
| JPEG-LS | 8.66 | 6.26 | 5.78 |
| JPEG XL lossless (effort 7) | 6.60 | 6.08 | 5.60 |
| A | 6.76 | 7.24 | 7.36 |
| B | 6.77 | 7.18 | 7.34 |
| C | 6.78 | 7.28 | 7.33 |
| D | 6.51 | 6.27 | 5.86 |
| **E (proposed)** | **6.49** (CR 2.46) | **6.17** (CR 2.59) | **5.76** (CR 2.78) |
| F | 6.52 | 6.18 | 5.76 |
| G | 6.52 | 6.13 | 5.77 |

## Method, file by file

* `wavecomp/lifting.py`: integer-to-integer 2-D lifting wavelet. Predict filters are fitted
  to each image, level and direction (IRLS on the L1 norm of the detail coefficients),
  quantised to 1/64 and stored in the header. A closed-loop check falls back to LeGall 5/3
  wherever that is better. The transform is exactly invertible.
* `wavecomp/fbse.py`: Fourier-Bessel series expansion of image rows and columns, then
  empirical-wavelet boundary detection on the FBSE spectrum. The first boundary sets the
  number of decomposition levels.
* `wavecomp/model.py`: neural entropy model. A modality-aware context encoder with a
  squeeze-and-excitation gate, a ConvLSTM state carried along the coarse-to-fine subband
  sequence, and 2x2 polyphase causal contexts. It predicts a discretised Laplace
  distribution for each coefficient.
* `wavecomp/codec.py`: reversible histogram packing, range coding (`constriction`), the
  stream header with a CRC-32 of the original pixels, the static-Laplace reference coder,
  and encode/decode.
* `wavecomp/dicomio.py`: stores a stream as encapsulated DICOM Pixel Data. It uses a
  private transfer syntax UID, so standard viewers cannot decode it.

## Data

All data is public and downloaded from the NCI Imaging Data Commons by
`scripts/fetch_idc.py`. The script keeps the stored DICOM pixel values (no rescale, no
windowing) and splits by patient with a fixed seed.

| Modality | Collection | Licence | Train / test patients | Test images |
|---|---|---|---|---|
| Chest X-ray (CR/DX, 10–15-bit) | LIDC-IDRI, doi:10.7937/K9/TCIA.2015.LO9QL9SX | CC BY 3.0 | 80 / 40 | 40 |
| Thoracic CT (12/16-bit) | LIDC-IDRI | CC BY 3.0 | 30 / 10 | 80 slices |
| Brain MRI (T1, T1-Gd, T2, FLAIR) | UPENN-GBM, doi:10.7937/TCIA.709X-DN49 | CC BY 4.0 | 24 / 8 | 138 slices |

The download needs about 1 GB of disk and takes about an hour.

## Reproduce

```bash
pip install -r requirements.txt
python scripts/fetch_idc.py --out data
python scripts/train.py --data data --out models/full.pt                               # optional: trained models are included
python scripts/train.py --data data --out models/plain.pt --no-lstm --no-attention     # optional
python scripts/evaluate.py --data data --models models --out results --verify DEFG
python scripts/summarize.py --results results
python scripts/make_figures.py --data data --models models --out results
python -m pytest -q tests
```

The experiments were run on a 4-core CPU without a GPU. Training takes about 50 minutes per
model, and the full evaluation takes about 1.5 hours.

## Notes

* The decoder recomputes the network's probabilities, so it must reproduce the encoder's
  floating-point outputs. Parameters are quantised before coding and the tests decode in a
  fresh process. Decoding on a different CPU or library version could, rarely, fail; the
  CRC check reports it if so.
* The DICOM wrapper uses a private transfer syntax and is not a standard DICOM codec.

## License

MIT (see `LICENSE`). The datasets keep their own licences, listed above.
