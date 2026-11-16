"""Download the public evaluation images from the NCI Imaging Data Commons (IDC).

Three modalities, all openly licensed DICOM (no registration needed):

  ct    LIDC-IDRI thoracic CT                  (CC BY 3.0, doi:10.7937/K9/TCIA.2015.LO9QL9SX)
  xray  LIDC-IDRI chest radiographs (CR + DX)  (same collection and licence)
  mri   UPENN-GBM brain MRI, BraTS-style       (CC BY 4.0, doi:10.7937/TCIA.709X-DN49)
        pre-processed T1, T1-Gd, T2, FLAIR

The stored DICOM pixel values (before any rescale slope/intercept or windowing) are
what a PACS archive must keep bit-exact, so those are what we save. Each series is
downloaded, sampled, written as a compressed .npz and the DICOM files are deleted,
so the disk footprint stays small.

Selection is deterministic (sorted patient IDs + fixed seed) and the split into
train / test is by patient, so no patient appears on both sides.

usage: python scripts/fetch_idc.py --out data [--modalities ct xray mri]
"""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pydicom
from idc_index import index

SEED = 20261104

PLAN = {
    # modality: (collection, filter, n_train_patients, n_test_patients, slices_per_series)
    "ct": dict(collection="lidc_idri", modalities=["CT"], n_train=30, n_test=10, slices=8),
    "xray": dict(collection="lidc_idri", modalities=["CR", "DX"], n_train=80, n_test=40, slices=1),
    "mri": dict(collection="upenn_gbm", modalities=["MR"], n_train=24, n_test=8, slices=6,
                descriptions=["t1 axial: Processed_CaPTk", "t1 axial stealth-post : Processed_CaPTk",
                              "Axial T2 tse: Processed_CaPTk", "t2_Flair_axial: Processed_CaPTk"]),
}


def select_series(df, cfg, rng):
    d = df[(df.collection_id == cfg["collection"]) & (df.Modality.isin(cfg["modalities"]))]
    if "descriptions" in cfg:
        d = d[d.SeriesDescription.isin(cfg["descriptions"])]
    if cfg["slices"] > 1:
        d = d[d.instanceCount >= 40]
    patients = sorted(d.PatientID.unique())
    rng.shuffle(patients)
    n = cfg["n_train"] + cfg["n_test"]
    patients = patients[:n]
    split = {p: ("train" if i < cfg["n_train"] else "test") for i, p in enumerate(patients)}
    rows = []
    for p in patients:
        dp = d[d.PatientID == p].sort_values("SeriesInstanceUID")
        if "descriptions" in cfg:
            # one series per MRI contrast for this patient
            dp = dp.drop_duplicates("SeriesDescription")
        else:
            dp = dp.head(1)
        for _, r in dp.iterrows():
            rows.append((p, split[p], r.SeriesInstanceUID, r.SeriesDescription, r.Modality))
    return rows


def load_series(folder):
    files = sorted(Path(folder).rglob("*.dcm"))
    ds = [pydicom.dcmread(f) for f in files]
    ds = [x for x in ds if "PixelData" in x]
    if not ds:
        return None, None
    # order slices along the patient axis when available
    def key(x):
        ipp = getattr(x, "ImagePositionPatient", None)
        return float(ipp[2]) if ipp is not None else float(getattr(x, "InstanceNumber", 0) or 0)
    ds.sort(key=key)
    imgs = []
    for x in ds:
        a = x.pixel_array  # stored values, no rescale
        if a.ndim == 2:
            imgs.append(a)
    if not imgs:
        return None, None
    x0 = ds[0]
    meta = dict(
        BitsAllocated=int(x0.BitsAllocated), BitsStored=int(x0.BitsStored),
        PixelRepresentation=int(x0.PixelRepresentation),
        PhotometricInterpretation=str(x0.PhotometricInterpretation),
        TransferSyntaxUID=str(x0.file_meta.TransferSyntaxUID),
        Manufacturer=str(getattr(x0, "Manufacturer", "")),
    )
    return imgs, meta


def pick_slices(imgs, k, rng):
    if k == 1:
        return [imgs[0]], [0]
    n = len(imgs)
    lo, hi = int(0.2 * n), int(0.8 * n)  # skip the ends of the scan (mostly air / background)
    idx = np.linspace(lo, hi - 1, k).round().astype(int)
    return [imgs[i] for i in idx], idx.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--modalities", nargs="+", default=list(PLAN))
    args = ap.parse_args()
    out = Path(args.out)
    client = index.IDCClient()
    df = client.index
    for mod in args.modalities:
        cfg = PLAN[mod]
        rng = np.random.default_rng(SEED)
        rows = select_series(df, cfg, rng)
        mdir = out / mod
        mdir.mkdir(parents=True, exist_ok=True)
        manifest = []
        for i, (pid, split, suid, desc, modality) in enumerate(rows):
            target = mdir / f"{split}_{pid}_{suid[-12:]}.npz"
            if target.exists():
                manifest.append(json.loads(target.with_suffix(".json").read_text()))
                continue
            tmp = Path(tempfile.mkdtemp(prefix="idc_"))
            try:
                client.download_from_selection(downloadDir=str(tmp), seriesInstanceUID=[suid],
                                               show_progress_bar=False, quiet=True)
                imgs, meta = load_series(tmp)
                if imgs is None:
                    print("skip (no 2-D pixel data)", suid)
                    continue
                sel, idx = pick_slices(imgs, cfg["slices"], rng)
                shapes = {a.shape for a in sel}
                if len(shapes) != 1:
                    sel = [sel[0]]
                    idx = idx[:1]
                arr = np.stack(sel)
                np.savez_compressed(target, images=arr)
                rec = dict(file=target.name, modality=mod, dicom_modality=modality, patient=pid,
                           split=split, series_uid=suid, description=desc, slice_index=idx,
                           shape=list(arr.shape), dtype=str(arr.dtype),
                           min=int(arr.min()), max=int(arr.max()), **meta)
                target.with_suffix(".json").write_text(json.dumps(rec))
                manifest.append(rec)
                print(f"[{mod} {i + 1}/{len(rows)}] {split} {pid} {arr.shape} {arr.dtype} "
                      f"bits={meta['BitsStored']} range=[{arr.min()},{arr.max()}]", flush=True)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        (mdir / "manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
