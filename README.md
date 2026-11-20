# medical-wavelet-codec

Code for *"AI-Enhanced Lossless Compression of Medical Images Using Adaptive Wavelet
Transforms and Neural Entropy Coding"* (M. Vivekanandan, R. Manjith).

This repository contains the code (Python, plus one C program) that produces the paper's
results: the codec, the baselines, and the training, evaluation and table/figure scripts.
Every reported rate is the size of a real, decodable bit stream, headers included, and every
test image is decoded again and checked bit-for-bit against the original.

## Which script produces which table and figure

| Paper | Content | Script | Output (generated) |
|---|---|---|---|
| Table 2 | Datasets, patients, train/test split | `scripts/fetch_idc.py` | `data/`, checked against `notebooks/data_manifest.json` |
| Table 3 | bpp per dataset and average, average CR, all methods | `scripts/evaluate.py`, then `scripts/manuscript_tables.py` | `results/manuscript/table3_bpp_cr.md` |
| Table 4 | Ablation of each component (configurations A–I) | same | `results/manuscript/table4_ablation.md` |
| Table 5 | Encode / decode throughput (MP/s) | same | `results/manuscript/table5_throughput.md` |
| Table 6 | Per-image statistics and paired tests | same | `results/manuscript/table6_stats.md` |
| Tables 7, 8 | Values quoted from other papers; qualitative comparison | none | — |
| Figure 4 | Average bpp per method | `scripts/manuscript_tables.py` | `results/manuscript/fig4_avg_bpp.png` |
| Figure 5 | Average compression ratio per method | same | `results/manuscript/fig5_avg_cr.png` |
| Figure 6 | bpp per method and dataset | same | `results/manuscript/fig6_bpp_by_dataset.png` |
| Figure 7 | Compression ratio per method and dataset | same | `results/manuscript/fig7_cr_by_dataset.png` |
| Section 5.8 | DICOM / PACS round trip (Orthanc, dcm4chee) | `scripts/pacs_test.py` | `results/pacs/` |
| — | Entropy models: D, E–H, I (proposed) | `scripts/train.py` | `models/plain.pt`, `models/full.pt`, `models/dmol.pt` |
| — | L3C baseline model (150,000 GPU steps) | `baselines/l3c/l3c.py`, `notebooks/train_l3c_colab.ipynb` | `models/l3c.pt` |
| — | Lossless check and FBSE illustration | `scripts/make_figures.py` | `results/fig_lossless.png`, `results/fig_fbse.png` |

`models/full.pt` and `models/plain.pt` are included. `models/dmol.pt` and `models/l3c.pt` are
not; train them with the commands below. `results/` holds the measured per-image results of
the first release (`per_image.csv`, configurations A–G and the imagecodecs baselines) and of
the L3C baseline (`per_image_l3c.csv`).

Configurations evaluated by `scripts/evaluate.py`:

| | Transform | Depth | Entropy model | Coder |
|---|---|---|---|---|
| A | LeGall 5/3 integer DWT | 5 levels | static Laplace per subband | range |
| B | adaptive lifting | 5 levels | static Laplace per subband | range |
| C | adaptive lifting | FBSE-chosen | static Laplace per subband | range |
| D | adaptive lifting | FBSE-chosen | neural context model, no ConvLSTM, no attention | range |
| E | adaptive lifting | FBSE-chosen | ConvLSTM + modality attention, Laplace | range |
| F | LeGall 5/3 | 5 levels | as E | range |
| G | adaptive lifting | 5 levels | as E | range |
| H | as E | as E | as E | rANS |
| I (proposed) | as E | as E | ConvLSTM + modality attention, DMOL (K = 5) | rANS |

Baselines: PNG, JPEG-LS, JPEG 2000 and JPEG XL through `imagecodecs`; CALIC
(`baselines/calic/calic.c`, our C implementation of the published algorithm for 8–16-bit
images); FLIF (reference encoder, built separately); L3C (`baselines/l3c/l3c.py`, the L3C
architecture reimplemented for one channel and any bit depth, because the official code and
weights are 8-bit RGB only).

## Method, file by file

* `wavecomp/lifting.py`: integer-to-integer 2-D lifting wavelet. Predict filters are fitted
  to each image, level and direction (IRLS on the L1 norm of the detail coefficients),
  quantised to 1/64 and stored in the header. A closed-loop check falls back to LeGall 5/3
  wherever that is better. The transform is exactly invertible.
* `wavecomp/fbse.py`: Fourier-Bessel series expansion of image rows and columns, then
  empirical-wavelet boundary detection on the FBSE spectrum; the first boundary sets the
  number of decomposition levels.
* `wavecomp/model.py`: neural entropy model. A modality-aware context encoder with a
  squeeze-and-excitation gate, a ConvLSTM state carried along the coarse-to-fine subband
  sequence, and 2x2 polyphase causal contexts. It predicts a discretised Laplace or a
  discretised mixture of logistics (DMOL) for each coefficient.
* `wavecomp/codec.py`: reversible histogram packing, range coding (`constriction`) or rANS,
  the stream header with a CRC-32 of the original pixels, the static-Laplace reference coder,
  and encode/decode.
* `wavecomp/rans.py`: interleaved rANS coder in NumPy for any alphabet size.
* `wavecomp/dicomio.py`: stores a stream as encapsulated DICOM Pixel Data with a private
  transfer syntax UID (standard viewers cannot decode it).

## Data

All data is public and downloaded from the NCI Imaging Data Commons by
`scripts/fetch_idc.py`, which keeps the stored DICOM pixel values (no rescale, no windowing)
and splits by patient with a fixed seed. `notebooks/data_manifest.json` lists every file
with a SHA-256 of its pixels, so a new download can be checked against the paper's data.

| Modality | Collection | Licence | Train / test patients | Test images |
|---|---|---|---|---|
| Chest X-ray (CR/DX, 10–15-bit) | LIDC-IDRI, doi:10.7937/K9/TCIA.2015.LO9QL9SX | CC BY 3.0 | 80 / 40 | 40 |
| Thoracic CT (12/16-bit) | LIDC-IDRI | CC BY 3.0 | 30 / 10 | 80 slices |
| Brain MRI (T1, T1-Gd, T2, FLAIR) | UPENN-GBM, doi:10.7937/TCIA.709X-DN49 | CC BY 4.0 | 24 / 8 | 138 slices |

## Reproduce

```bash
pip install -r requirements.txt
gcc -O2 -o baselines/calic/calic baselines/calic/calic.c            # CALIC baseline
python scripts/fetch_idc.py --out data                               # about 20 min
python scripts/train.py --data data --out models/full.pt                              # E-H
python scripts/train.py --data data --out models/plain.pt --no-lstm --no-attention    # D
python scripts/train.py --data data --out models/dmol.pt --mixture 5                  # I (proposed)
python baselines/l3c/l3c.py train --data data --out models/l3c.pt --steps 150000 --batch 16 --lr 1e-4   # GPU
python scripts/evaluate.py --data data --models models --out results \
    --configs A B C D E F G H I png jpegls jpeg2000 jpegxl calic flif l3c --verify DEFGHI
python scripts/manuscript_tables.py --results results                 # Tables 3-6, Figures 4-7
python -m pytest -q tests
```

The proposed models were trained, and every codec was timed, on a 4-core CPU without a GPU
(about 50 minutes per model). The L3C baseline needs a GPU: `notebooks/train_l3c_colab.ipynb`
clones this repository on Google Colab, trains L3C with resumable checkpoints in Google Drive
and runs the bit-exact L3C evaluation. FLIF must be built from github.com/FLIF-hub/FLIF and
be on `PATH`.

## Notes

* The decoder recomputes the network's probabilities, so it must reproduce the encoder's
  floating-point outputs. Parameters are quantised before coding and the tests decode in a
  fresh process. Decoding on a different CPU or library version could, rarely, fail; the
  CRC check reports it if so.
* The DICOM wrapper uses a private transfer syntax and is not a standard DICOM codec.

## License

MIT (see `LICENSE`). The datasets keep their own licences, listed above.
