# Results (measured)

All rates are real bit-stream sizes including headers, pooled over every pixel of the held-out test patients. CR = 16 / bpp (DICOM stores 16 bits per pixel). Every stream listed with a check count was decoded and compared bit-for-bit with the original.

## Chest X-ray (LIDC-IDRI CR/DX)

40 images from 40 test patients (of 40 attempted), size 2022x2022 median, Bits Stored 10, 12, 14, 15.

| Codec | bpp | CR | enc s/img | dec s/img | decoded & verified |
|---|---:|---:|---:|---:|---:|
| PNG (zlib, level 9) | 11.142 | 1.44 | 0.73 | 0.09 | 40/40 |
| JPEG 2000 lossless (OpenJPEG, 5/3) | 8.816 | 1.81 | 1.09 | 0.94 | 40/40 |
| JPEG-LS (CharLS) | 8.659 | 1.85 | 0.18 | 0.16 | 40/40 |
| JPEG XL lossless (libjxl, effort 7) | 6.601 | 2.42 | 3.37 | 0.67 | 40/40 |
| A  LeGall 5/3 DWT (5 lv) + static Laplace | 6.757 | 2.37 | 0.89 | 1.18 | 40/40 |
| B  adaptive lifting (5 lv) + static Laplace | 6.771 | 2.36 | 2.85 | 1.22 | 40/40 |
| C  adaptive lifting, FBSE levels + static Laplace | 6.781 | 2.36 | 2.66 | 1.26 | 40/40 |
| D  C + neural context model (no ConvLSTM, no attention) | 6.505 | 2.46 | 10.90 | 10.19 | 40/40 |
| E  C + ConvLSTM neural entropy model (proposed) | 6.493 | 2.46 | 12.99 | 11.99 | 40/40 |
| F  LeGall 5/3 (5 lv) + ConvLSTM neural model | 6.524 | 2.45 | 12.77 | 12.80 | 40/40 |
| G  adaptive lifting (5 lv) + ConvLSTM neural model | 6.519 | 2.45 | 13.62 | 11.99 | 40/40 |

## Thoracic CT (LIDC-IDRI)

80 images from 10 test patients (of 80 attempted), size 512x512 median, Bits Stored 12, 16.

| Codec | bpp | CR | enc s/img | dec s/img | decoded & verified |
|---|---:|---:|---:|---:|---:|
| PNG (zlib, level 9) | 8.675 | 1.84 | 0.04 | 0.01 | 80/80 |
| JPEG 2000 lossless (OpenJPEG, 5/3) | 6.471 | 2.47 | 0.05 | 0.05 | 80/80 |
| JPEG-LS (CharLS) | 6.255 | 2.56 | 0.01 | 0.01 | 80/80 |
| JPEG XL lossless (libjxl, effort 7) | 6.083 | 2.63 | 0.18 | 0.04 | 80/80 |
| A  LeGall 5/3 DWT (5 lv) + static Laplace | 7.240 | 2.21 | 0.05 | 0.08 | 80/80 |
| B  adaptive lifting (5 lv) + static Laplace | 7.177 | 2.23 | 0.16 | 0.05 | 80/80 |
| C  adaptive lifting, FBSE levels + static Laplace | 7.283 | 2.20 | 0.17 | 0.06 | 80/80 |
| D  C + neural context model (no ConvLSTM, no attention) | 6.267 | 2.55 | 0.49 | 0.41 | 80/80 |
| E  C + ConvLSTM neural entropy model (proposed) | 6.167 | 2.59 | 0.56 | 0.47 | 80/80 |
| F  LeGall 5/3 (5 lv) + ConvLSTM neural model | 6.178 | 2.59 | 0.47 | 0.49 | 80/80 |
| G  adaptive lifting (5 lv) + ConvLSTM neural model | 6.130 | 2.61 | 0.59 | 0.50 | 80/80 |

## Brain MRI (UPENN-GBM)

138 images from 8 test patients (of 138 attempted), size 256x192 median, Bits Stored 16.

| Codec | bpp | CR | enc s/img | dec s/img | decoded & verified |
|---|---:|---:|---:|---:|---:|
| PNG (zlib, level 9) | 8.017 | 2.00 | 0.01 | 0.00 | 138/138 |
| JPEG 2000 lossless (OpenJPEG, 5/3) | 6.016 | 2.66 | 0.01 | 0.01 | 138/138 |
| JPEG-LS (CharLS) | 5.784 | 2.77 | 0.00 | 0.00 | 138/138 |
| JPEG XL lossless (libjxl, effort 7) | 5.603 | 2.86 | 0.04 | 0.01 | 138/138 |
| A  LeGall 5/3 DWT (5 lv) + static Laplace | 7.358 | 2.17 | 0.01 | 0.02 | 138/138 |
| B  adaptive lifting (5 lv) + static Laplace | 7.338 | 2.18 | 0.06 | 0.02 | 138/138 |
| C  adaptive lifting, FBSE levels + static Laplace | 7.330 | 2.18 | 0.04 | 0.01 | 138/138 |
| D  C + neural context model (no ConvLSTM, no attention) | 5.857 | 2.73 | 0.14 | 0.12 | 138/138 |
| E  C + ConvLSTM neural entropy model (proposed) | 5.757 | 2.78 | 0.22 | 0.13 | 138/138 |
| F  LeGall 5/3 (5 lv) + ConvLSTM neural model | 5.763 | 2.78 | 0.13 | 0.13 | 138/138 |
| G  adaptive lifting (5 lv) + ConvLSTM neural model | 5.769 | 2.77 | 0.15 | 0.13 | 138/138 |
