"""PACS round trip: store compressed DICOM files in a real archive over the DICOM
network and retrieve them again, then decode and compare pixel by pixel.

For every test image:
  1. encode with the proposed codec (configuration E) and wrap the stream in a DICOM
     file with the codec's private transfer syntax (wavecomp/dicomio.py);
  2. C-STORE it to each archive (pynetdicom, presentation context = the private syntax);
  3. C-GET it back over a new association that offers only the private syntax, so the
     archive must return the stored bytes rather than transcode them;
  4. decode the returned Pixel Data and compare with the original pixels.

The archives are passed as AET@host:port (DICOM network), or as an http:// URL for
Orthanc's REST API (POST /instances, GET /instances/{id}/file). Tested with Orthanc 1.12.2 (Ubuntu package)
and dcm4chee-arc-light 5.36.0 (official Docker images, PostgreSQL + OpenLDAP).

usage: python scripts/pacs_test.py --data data --models models --out results/pacs \
           --pacs ORTHANC@localhost:4242 DCM4CHEE@localhost:11112
"""
import argparse
import csv
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pydicom
import torch
from pydicom.dataset import Dataset
from pydicom.encaps import generate_frames
from pydicom.uid import generate_uid
from pynetdicom import AE, StoragePresentationContexts, build_role, evt  # noqa: F401
from pynetdicom.sop_class import (PatientRootQueryRetrieveInformationModelGet,  # noqa: E402
                                  SecondaryCaptureImageStorage)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wavecomp import codec, dicomio  # noqa: E402
from wavecomp.model import MODALITIES  # noqa: E402

TS = dicomio.WVC1_TRANSFER_SYNTAX
# tell pydicom/pynetdicom the private syntax is an encapsulated Explicit VR Little Endian one
pydicom.uid.register_transfer_syntax(TS, implicit_vr=False, little_endian=True)


def _patch(name, value):
    """pynetdicom builds fresh UID objects for negotiated syntaxes, which lose the private
    registration; answer for the private UID directly."""
    orig = getattr(pydicom.uid.UID, name).fget
    setattr(pydicom.uid.UID, name, property(lambda self: value if str(self) == str(TS) else orig(self)))


for _n, _v in [("is_implicit_VR", False), ("is_little_endian", True), ("is_transfer_syntax", True),
               ("is_encapsulated", True), ("is_compressed", True), ("is_deflated", False)]:
    _patch(_n, _v)


def parse(spec):
    aet, hp = spec.split("@")
    host, port = hp.rsplit(":", 1)
    return aet, host, int(port)


def c_store(spec, path):
    aet, host, port = parse(spec)
    ae = AE(ae_title="WVC_TEST")
    ae.add_requested_context(SecondaryCaptureImageStorage, [TS])
    assoc = ae.associate(host, port, ae_title=aet)
    if not assoc.is_established:
        return "rejected: no presentation context accepted (private transfer syntax)"
    try:
        ctx = [c for c in assoc.accepted_contexts if c.abstract_syntax == SecondaryCaptureImageStorage]
        if not ctx:
            return "private transfer syntax not accepted"
        st = assoc.send_c_store(pydicom.dcmread(path))
        return f"0x{st.Status:04X}" if st else "no response"
    finally:
        assoc.release()


def c_get(spec, ds_sent):
    aet, host, port = parse(spec)
    got = []

    def on_store(event):
        ds = event.dataset
        ds.file_meta = event.file_meta
        got.append(ds)
        return 0x0000

    ae = AE(ae_title="WVC_TEST")
    ae.add_requested_context(PatientRootQueryRetrieveInformationModelGet)
    ae.add_requested_context(SecondaryCaptureImageStorage, [TS])
    role = build_role(SecondaryCaptureImageStorage, scp_role=True)
    assoc = ae.associate(host, port, ae_title=aet, ext_neg=[role], evt_handlers=[(evt.EVT_C_STORE, on_store)])
    if not assoc.is_established:
        return None, "association rejected"
    q = Dataset()
    q.QueryRetrieveLevel = "IMAGE"
    q.PatientID = ds_sent.PatientID
    q.StudyInstanceUID = ds_sent.StudyInstanceUID
    q.SeriesInstanceUID = ds_sent.SeriesInstanceUID
    q.SOPInstanceUID = ds_sent.SOPInstanceUID
    last = None
    try:
        for status, _ in assoc.send_c_get(q, PatientRootQueryRetrieveInformationModelGet):
            if status:
                last = status.Status
    finally:
        assoc.release()
    return (got[0] if got else None), (f"0x{last:04X}" if last is not None else "no response")


def rest_roundtrip(url, path):
    """Orthanc REST API: POST the file to /instances, GET it back from /instances/{id}/file."""
    import io
    import urllib.request
    req = urllib.request.Request(url.rstrip("/") + "/instances", data=Path(path).read_bytes(), method="POST")
    with urllib.request.urlopen(req) as r:
        info = json.loads(r.read())
    if info.get("Status") not in ("Success", "AlreadyStored"):
        return None, str(info.get("Status")), "no instance"
    with urllib.request.urlopen(url.rstrip("/") + f"/instances/{info['ID']}/file") as r:
        ds = pydicom.dcmread(io.BytesIO(r.read()))
    return ds, info["Status"], "HTTP 200"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--models", default="models")
    ap.add_argument("--out", default="results/pacs")
    ap.add_argument("--pacs", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    model = codec.load_model(Path(args.models) / "full.pt")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "pacs_roundtrip.csv"
    done = set()
    if rows_path.exists():
        with open(rows_path) as f:
            done = {(r["image"], r["pacs"]) for r in csv.DictReader(f)}
    fcsv = open(rows_path, "a", newline="")
    w = csv.writer(fcsv)
    if not done:
        w.writerow(["modality", "image", "pacs", "file_bytes", "store_status", "get_status",
                    "returned_ts", "bytes_identical", "pixels_identical", "seconds"])
    for mod in MODALITIES:
        items = []
        for f in sorted((Path(args.data) / mod).glob("test_*.npz")):
            meta = json.loads(f.with_suffix(".json").read_text())
            for k, im in enumerate(np.load(f)["images"]):
                items.append((f"{f.stem}#{k}", meta, im))
        if args.limit:
            items = items[:args.limit]
        for name, meta, img in items:
            if all((name, p) in done for p in args.pacs):
                continue
            blob = codec.encode(img, model, mod)
            with tempfile.TemporaryDirectory() as td:
                path = f"{td}/x.dcm"
                dicomio.write(path, blob, img)
                ds_sent = pydicom.dcmread(path)
                ds_sent.PatientID = meta["patient"]
                ds_sent.PatientName = meta["patient"]
                ds_sent.StudyInstanceUID = generate_uid()
                ds_sent.SeriesInstanceUID = generate_uid()
                ds_sent.Modality = {"xray": "DX", "ct": "CT", "mri": "MR"}[mod]
                ds_sent.SeriesNumber = 1
                ds_sent.InstanceNumber = 1
                ds_sent.save_as(path)
                ds_sent = pydicom.dcmread(path)
                nbytes = Path(path).stat().st_size
                for p in args.pacs:
                    if (name, p) in done:
                        continue
                    t = time.time()
                    if p.startswith("http"):
                        ds_back, s_store, s_get = rest_roundtrip(p, path)
                    else:
                        s_store = c_store(p, path)
                        ds_back, s_get = c_get(p, ds_sent) if s_store == "0x0000" else (None, "skipped")
                    ts_back = str(ds_back.file_meta.TransferSyntaxUID) if ds_back is not None else ""
                    same_bytes = pix_ok = ""
                    if ds_back is not None:
                        back = ds_back.PixelData
                        same_bytes = int(back == ds_sent.PixelData)
                        stream = next(generate_frames(back, number_of_frames=1))
                        try:
                            pix_ok = int(np.array_equal(codec.decode(stream, model), img))
                        except Exception as e:  # noqa: BLE001
                            pix_ok = f"decode failed: {e}"
                    w.writerow([mod, name, p, nbytes, s_store, s_get, ts_back, same_bytes, pix_ok,
                                f"{time.time() - t:.2f}"])
                    fcsv.flush()
            print(mod, name, flush=True)
    fcsv.close()


if __name__ == "__main__":
    main()
