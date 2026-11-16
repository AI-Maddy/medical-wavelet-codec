"""Wrap a WVC1 bit stream in a DICOM file (and get it back).

The compressed stream is stored as encapsulated Pixel Data (7FE0,0010) in a
single fragment, with the image attributes (Rows, Columns, Bits Stored, Pixel
Representation, Photometric Interpretation, ...) copied from the source so a
PACS can index and route the object normally.

Important limitation: the transfer syntax is a *private* UID.  Standard DICOM
viewers cannot decode it; a receiving system needs this codec (or the archive
transcodes to a standard transfer syntax on retrieval).  Making the format a
standard transfer syntax would require a DICOM Supplement.
"""
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.encaps import encapsulate
from pydicom.uid import UID, generate_uid

# Private transfer syntax UID under pydicom's example root (replace with an
# organisation root before any real deployment).
WVC1_TRANSFER_SYNTAX = UID("1.2.826.0.1.3680043.8.498.20261104.1")
SECONDARY_CAPTURE = UID("1.2.840.10008.5.1.4.1.1.7")

COPY = ["Modality", "Rows", "Columns", "BitsAllocated", "BitsStored", "HighBit",
        "PixelRepresentation", "PhotometricInterpretation", "SamplesPerPixel",
        "RescaleSlope", "RescaleIntercept", "PixelSpacing", "SliceThickness",
        "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID", "SOPClassUID"]


def write(path, blob, image, source=None):
    """Write blob (bytes from codec.encode) as a DICOM file.

    image  the original array (for Rows/Columns/bit depth if no source dataset)
    source optional pydicom Dataset whose image attributes are copied
    """
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = WVC1_TRANSFER_SYNTAX
    sop_class = getattr(source, "SOPClassUID", SECONDARY_CAPTURE) if source is not None else SECONDARY_CAPTURE
    sop_uid = getattr(source, "SOPInstanceUID", None) if source is not None else None
    meta.MediaStorageSOPClassUID = sop_class
    meta.MediaStorageSOPInstanceUID = sop_uid or generate_uid()
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    if source is not None:
        for k in COPY:
            if k in source:
                setattr(ds, k, source.data_element(k).value)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Rows, ds.Columns = image.shape
    ds.SamplesPerPixel = 1
    if "PhotometricInterpretation" not in ds:
        ds.PhotometricInterpretation = "MONOCHROME2"
    if "BitsAllocated" not in ds:
        ds.BitsAllocated = image.dtype.itemsize * 8
        ds.BitsStored = ds.BitsAllocated
        ds.HighBit = ds.BitsStored - 1
        ds.PixelRepresentation = int(image.dtype.kind == "i")
    ds.ImageComments = "Lossless WVC1 (adaptive integer wavelet + neural entropy coding)"
    ds.LossyImageCompression = "00"
    ds.PixelData = encapsulate([blob])
    ds["PixelData"].VR = "OB"
    ds["PixelData"].is_undefined_length = True
    ds.save_as(str(path), enforce_file_format=True, implicit_vr=False, little_endian=True)


def read(path):
    """Return the WVC1 bit stream stored in a file written by write()."""
    import pydicom
    from pydicom.encaps import generate_frames
    ds = pydicom.dcmread(str(path))
    if ds.file_meta.TransferSyntaxUID != WVC1_TRANSFER_SYNTAX:
        raise ValueError("not a WVC1-compressed DICOM file")
    frame = next(generate_frames(ds.PixelData, number_of_frames=1))
    return frame, ds    # may carry one trailing pad byte; the decoder ignores it
