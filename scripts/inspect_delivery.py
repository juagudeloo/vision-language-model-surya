#!/usr/bin/env python3
"""inspect_delivery.py — Report what a data delivery contains and whether it can feed Surya.

Walks a folder of FITS, Surya-format netCDF and metadata files, reads headers only (no pixel data
is loaded), and checks the delivery against what Surya's input requires: 13 channels, full-disk
4096x4096 frames, two instants about 60 minutes apart, and a known processing level. It also checks
that the metadata carries captions and region information, which training needs.

Usage:
  python scripts/inspect_delivery.py /path/to/delivery
  python scripts/inspect_delivery.py /path/to/delivery --json-out report.json
"""

import argparse
import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path

from astropy.io import fits
from pydantic import BaseModel, Field

SURYA_CHANNELS = [
    "aia94", "aia131", "aia171", "aia193", "aia211", "aia304", "aia335", "aia1600",
    "hmi_m", "hmi_bx", "hmi_by", "hmi_bz", "hmi_v",
]
AIA_WAVELENGTHS = {94, 131, 171, 193, 211, 304, 335, 1600}
FULL_DISK = (4096, 4096)
FITS_SUFFIXES = {".fits", ".fit", ".fts"}
METADATA_SUFFIXES = {".json", ".jsonl", ".csv", ".tsv", ".yaml", ".yml", ".txt", ".md", ".xlsx"}
CAPTION_KEYS = ("caption", "description", "text", "label")
REGION_KEYS = ("bbox", "x0", "x1", "y0", "y1", "center", "heliographic", "active_region", "noaa", "hpc", "tx", "ty")
VECTOR_HINTS = ("field", "inclination", "azimuth", "disambig", "bx", "by", "bz", "br", "bt", "bp")


class FileInfo(BaseModel):
    """What one delivered file turned out to be, read from its header only."""

    path: str = Field(description="Path relative to the delivery root.")
    kind: str = Field(description="fits, netcdf, metadata or other.")
    size_mb: float = Field(description="File size in megabytes.")
    product: str | None = Field(default=None, description="aia, hmi, suvi or other instrument, when known.")
    channel: str | None = Field(default=None, description="Surya channel name such as aia171 or hmi_m.")
    channels_present: list[str] = Field(default_factory=list, description="For netCDF: the variables it holds.")
    level: str | None = Field(default=None, description="Processing level from LVL_NUM, when the header has it.")
    t_obs: datetime | None = Field(default=None, description="Observation time, taken as UTC.")
    shape: tuple[int, int] | None = Field(default=None, description="Image height and width.")
    cdelt_arcsec: float | None = Field(default=None, description="Pixel scale in arcseconds.")
    rotation_deg: float | None = Field(default=None, description="Rotation of the image axes from solar north.")
    exptime: float | None = Field(default=None, description="Exposure time in seconds.")
    state: str = Field(default="unknown", description="Processing state, see classify_state.")
    notes: list[str] = Field(default_factory=list, description="Things worth a human's attention for this file.")


class MetadataInfo(BaseModel):
    """What a metadata file holds and whether it looks usable for training."""

    path: str = Field(description="Path relative to the delivery root.")
    format: str = Field(description="File format.")
    n_records: int | None = Field(default=None, description="Rows or top-level entries, when countable.")
    fields: list[str] = Field(default_factory=list, description="Column or key names found.")
    caption_fields: list[str] = Field(default_factory=list, description="Fields that look like captions.")
    region_fields: list[str] = Field(default_factory=list, description="Fields that look like region information.")
    n_nonempty_captions: int | None = Field(default=None, description="Records with a non-empty caption.")
    fits_linked: int = Field(default=0, description="Delivered FITS file names that this file mentions.")


class Instant(BaseModel):
    """Files observed at about the same time, grouped to see which channels are covered."""

    t: datetime = Field(description="Earliest observation time in the group.")
    channels: list[str] = Field(description="Surya channels covered.")
    missing: list[str] = Field(description="Surya channels not covered.")
    n_files: int = Field(description="Files in the group.")


class DeliveryReport(BaseModel):
    """Everything found in a delivery plus a verdict per Surya requirement."""

    root: str = Field(description="Delivery folder that was scanned.")
    counts: dict[str, int] = Field(description="Files by kind.")
    files: list[FileInfo] = Field(description="Per-file details for FITS and netCDF.")
    metadata: list[MetadataInfo] = Field(description="Per-file details for metadata files.")
    instants: list[Instant] = Field(description="Observation instants found.")
    pairs_60min: int = Field(description="Instant pairs about 60 minutes apart.")
    complete_pairs_60min: int = Field(description="Of those, pairs where both instants cover all 13 channels.")
    verdict: dict[str, str] = Field(description="Requirement name to a one-line verdict.")


def parse_obs_time(value: object) -> datetime | None:
    """Parse the time formats that SDO, HMI and SUVI headers use.

    Args:
        value: A header value such as '2023-03-15T09:30:09.63Z' or '2014.01.07_14:59:51.661_TAI'.

    Returns:
        A naive datetime taken as UTC, or None when it cannot be parsed. TAI is treated as UTC,
        which is off by under a minute and does not matter for grouping instants.
    """
    if not isinstance(value, str):
        return None
    text = value.strip().rstrip("Z")
    match = re.fullmatch(r"(\d{4})\.(\d{2})\.(\d{2})_(\d{2}):(\d{2}):(\d{2})(\.\d+)?(_TAI|_UTC)?", text)
    if match:
        y, mo, d, h, mi, s = (int(match.group(i)) for i in range(1, 7))
        return datetime(y, mo, d, h, mi, s)
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def image_hdu(hdul: fits.HDUList):
    """Return the first HDU that holds a 2D image, including compressed ones.

    Args:
        hdul: An open FITS file.

    Returns:
        The image HDU, or None when the file has no 2D image.
    """
    for hdu in hdul:
        shape = getattr(hdu, "shape", None)
        if shape is not None and len(shape) == 2:
            return hdu
    return None


def rotation_from_header(header: fits.Header) -> float | None:
    """Rotation of the image axes in degrees, from CROTA2 or the PC matrix.

    Args:
        header: The image header.

    Returns:
        Degrees, or None when the header carries neither form.
    """
    if "CROTA2" in header:
        return float(header["CROTA2"])
    if "PC2_1" in header and "PC1_1" in header:
        return math.degrees(math.atan2(float(header["PC2_1"]), float(header["PC1_1"])))
    return None


def classify_state(info: FileInfo) -> str:
    """Name the processing state of a file, which says how much work it needs before Surya.

    Args:
        info: A file with product, level, shape, scale and rotation already filled in.

    Returns:
        One of surya_netcdf, aia_level_1_5, aia_level_1, hmi_native, hmi_other, cutout,
        other_instrument or unknown.
    """
    if info.kind == "netcdf":
        return "surya_netcdf" if info.shape == FULL_DISK else "cutout"
    if info.product not in ("aia", "hmi"):
        return "other_instrument" if info.product else "unknown"
    if info.shape != FULL_DISK:
        return "cutout"
    north_up = info.rotation_deg is not None and abs(info.rotation_deg) < 1e-4
    on_grid = info.cdelt_arcsec is not None and abs(info.cdelt_arcsec - 0.6) < 1e-4
    if info.product == "aia":
        if info.level is not None:
            return "aia_level_1_5" if info.level.startswith("1.5") else "aia_level_1"
        return "aia_level_1_5" if (north_up and on_grid) else "aia_level_1"
    return "hmi_native" if not (north_up and on_grid) else "hmi_other"


def inspect_fits(path: Path, root: Path) -> FileInfo:
    """Read one FITS file's header and describe it.

    Args:
        path: The FITS file.
        root: The delivery root, used to print a relative path.

    Returns:
        A FileInfo. Unreadable files come back with a note instead of raising.
    """
    info = FileInfo(path=str(path.relative_to(root)), kind="fits", size_mb=round(path.stat().st_size / 1e6, 2))
    try:
        with fits.open(path, memmap=True) as hdul:
            hdu = image_hdu(hdul)
            if hdu is None:
                info.notes.append("no 2D image HDU")
                return info
            header = hdu.header
            primary = hdul[0].header
            info.shape = (int(hdu.shape[0]), int(hdu.shape[1]))
            telescope = str(header.get("TELESCOP", primary.get("TELESCOP", ""))).upper()
            instrument = str(header.get("INSTRUME", primary.get("INSTRUME", ""))).upper()
            if "AIA" in instrument or "AIA" in telescope:
                info.product = "aia"
                wavelength = header.get("WAVELNTH")
                if wavelength is not None and int(wavelength) in AIA_WAVELENGTHS:
                    info.channel = f"aia{int(wavelength)}"
                elif wavelength is not None:
                    info.notes.append(f"AIA {int(wavelength)} A is not one of Surya's channels")
            elif "HMI" in instrument or "HMI" in telescope:
                info.product = "hmi"
                content = str(header.get("CONTENT", "")).upper()
                name = path.name.lower()
                if "MAGNETOGRAM" in content or ".magnetogram" in name:
                    info.channel = "hmi_m"
                elif "DOPPLER" in content or "dopplergram" in name:
                    info.channel = "hmi_v"
                elif any(hint in re.split(r"[._\-]", name) for hint in VECTOR_HINTS):
                    info.notes.append("looks like a vector field product; needs conversion to Bx, By, Bz")
                else:
                    info.notes.append(f"HMI product {content.title() or 'of unknown content'} is not one of Surya's channels")
            elif "SUVI" in instrument or "GOES" in telescope:
                info.product = "suvi"
                info.notes.append("GOES SUVI, not SDO: Surya was not trained on it")
            level = header.get("LVL_NUM")
            info.level = str(level) if level is not None else None
            for key in ("T_OBS", "DATE-OBS", "T_REC"):
                info.t_obs = parse_obs_time(header.get(key))
                if info.t_obs:
                    break
            scale = header.get("CDELT1")
            if scale is not None:
                unit = str(header.get("CUNIT1", "arcsec")).lower()
                info.cdelt_arcsec = float(scale) * (3600.0 if unit.startswith("deg") else 1.0)
            info.rotation_deg = rotation_from_header(header)
            exptime = header.get("EXPTIME")
            info.exptime = float(exptime) if exptime is not None else None
    except Exception as exc:
        info.notes.append(f"could not read: {type(exc).__name__}: {exc}")
    info.state = classify_state(info)
    return info


def inspect_netcdf(path: Path, root: Path) -> FileInfo:
    """Describe a Surya-format netCDF file from its variables and attributes.

    Args:
        path: The netCDF file.
        root: The delivery root, used to print a relative path.

    Returns:
        A FileInfo with the variables it holds. Unreadable files come back with a note.
    """
    import xarray as xr

    info = FileInfo(path=str(path.relative_to(root)), kind="netcdf", size_mb=round(path.stat().st_size / 1e6, 2))
    try:
        with xr.open_dataset(path, engine="h5netcdf") as ds:
            info.channels_present = [name for name in SURYA_CHANNELS if name in ds.data_vars]
            first = next(iter(ds.data_vars.values()), None)
            if first is not None and first.ndim == 2:
                info.shape = (int(first.shape[0]), int(first.shape[1]))
            stamp = ds.attrs.get("data_time")
            if stamp:
                info.t_obs = datetime.strptime(str(stamp), "%Y%m%d_%H%M")
            info.level = "1.5" if "level-1.5" in str(ds.attrs.get("title", "")).lower() else None
            extra = [name for name in ds.data_vars if name not in SURYA_CHANNELS]
            if extra:
                info.notes.append(f"variables Surya does not use: {extra}")
    except Exception as exc:
        info.notes.append(f"could not read: {type(exc).__name__}: {exc}")
    info.state = classify_state(info)
    return info


def summarize_metadata(path: Path, root: Path, fits_names: list[str]) -> MetadataInfo:
    """Summarize a metadata file: its fields, captions, region information and links to FITS files.

    Args:
        path: The metadata file.
        root: The delivery root, used to print a relative path.
        fits_names: Base names of the delivered FITS files, to count how many the file mentions.

    Returns:
        A MetadataInfo. Formats that cannot be parsed still report the FITS names they mention.
    """
    info = MetadataInfo(path=str(path.relative_to(root)), format=path.suffix.lower().lstrip("."))
    records: list[dict] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace") if path.suffix.lower() != ".xlsx" else ""
        info.fits_linked = sum(1 for name in fits_names if name in text or Path(name).stem in text)
        suffix = path.suffix.lower()
        if suffix == ".json":
            data = json.loads(text)
            if isinstance(data, dict):
                inner = [v for v in data.values() if isinstance(v, list) and v and isinstance(v[0], dict)]
                records = inner[0] if inner else [data]
            elif isinstance(data, list):
                records = [r for r in data if isinstance(r, dict)]
        elif suffix == ".jsonl":
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
        elif suffix in (".csv", ".tsv"):
            delimiter = "\t" if suffix == ".tsv" else ","
            records = list(csv.DictReader(text.splitlines(), delimiter=delimiter))
        elif suffix == ".xlsx":
            info.fields = ["(xlsx is not parsed here; export it to csv to inspect its columns)"]
    except Exception as exc:
        info.fields = [f"could not parse: {type(exc).__name__}: {exc}"]
    if records:
        info.n_records = len(records)
        keys = sorted({str(k) for record in records for k in record})
        info.fields = keys
        info.caption_fields = [k for k in keys if any(c in k.lower() for c in CAPTION_KEYS)]
        info.region_fields = [k for k in keys if any(r in k.lower() for r in REGION_KEYS)]
        if info.caption_fields:
            field = info.caption_fields[0]
            info.n_nonempty_captions = sum(1 for r in records if str(r.get(field) or "").strip())
    return info


def group_instants(files: list[FileInfo], tolerance_min: float) -> list[Instant]:
    """Group SDO files observed within a few minutes of each other into instants.

    Args:
        files: Inspected files. Only AIA, HMI and Surya netCDF files with a time are used.
        tolerance_min: Largest gap, from the first file of a group, to still count as the same instant.

    Returns:
        Instants in time order, each with the channels it covers and the ones it lacks.
    """
    usable = [f for f in files if f.t_obs and (f.kind == "netcdf" or f.channel)]
    usable.sort(key=lambda f: f.t_obs)
    groups: list[list[FileInfo]] = []
    for f in usable:
        if groups and (f.t_obs - groups[-1][0].t_obs).total_seconds() / 60 <= tolerance_min:
            groups[-1].append(f)
        else:
            groups.append([f])
    instants = []
    for group in groups:
        channels = {c for f in group for c in (f.channels_present or ([f.channel] if f.channel else []))}
        covered = [c for c in SURYA_CHANNELS if c in channels]
        instants.append(Instant(
            t=group[0].t_obs, channels=covered, missing=[c for c in SURYA_CHANNELS if c not in channels], n_files=len(group),
        ))
    return instants


def count_pairs(instants: list[Instant], gap_min: float, tolerance_min: float) -> tuple[int, int]:
    """Count pairs of instants separated by about gap_min minutes.

    Args:
        instants: Instants in time order.
        gap_min: Wanted separation in minutes, 60 for Surya.
        tolerance_min: Allowed deviation from gap_min.

    Returns:
        Total pairs, and how many of them have all 13 channels in both instants.
    """
    total = complete = 0
    for i, a in enumerate(instants):
        for b in instants[i + 1:]:
            gap = (b.t - a.t).total_seconds() / 60
            if abs(gap - gap_min) <= tolerance_min:
                total += 1
                complete += int(not a.missing and not b.missing)
    return total, complete


def assess(files: list[FileInfo], metadata: list[MetadataInfo], instants: list[Instant], pairs: int, complete: int) -> dict[str, str]:
    """Judge the delivery against each of Surya's requirements and the training needs.

    Args:
        files: Inspected FITS and netCDF files.
        metadata: Inspected metadata files.
        instants: Observation instants.
        pairs: Instant pairs about 60 minutes apart.
        complete: Of those, pairs with all 13 channels on both sides.

    Returns:
        A requirement name mapped to a one-line verdict starting with OK, WARN or FAIL.
    """
    sdo = [f for f in files if f.kind == "netcdf" or f.product in ("aia", "hmi")]
    states = {s: sum(1 for f in sdo if f.state == s) for s in {f.state for f in sdo}}
    verdict: dict[str, str] = {}
    if not sdo:
        verdict["files"] = "FAIL: no SDO files (AIA, HMI or Surya netCDF) found"
    cutouts = states.get("cutout", 0)
    verdict["full_disk_4096"] = (
        "OK: all SDO files are 4096x4096" if sdo and cutouts == 0
        else f"FAIL: {cutouts} of {len(sdo)} SDO files are cutouts; Surya needs the full disk" if sdo else "FAIL: nothing to check"
    )
    best = max((len(i.channels) for i in instants), default=0)
    verdict["channels_13"] = (
        "OK: at least one instant covers all 13 channels" if best == 13
        else f"FAIL: the best instant covers {best} of 13 channels" if instants else "FAIL: no instants found"
    )
    verdict["two_instants_60min"] = (
        f"OK: {complete} pair(s) about 60 minutes apart with all channels" if complete
        else f"FAIL: {pairs} pair(s) about 60 minutes apart, none with all 13 channels on both sides"
    )
    needs = {s: n for s, n in states.items() if s in ("aia_level_1", "hmi_native")}
    verdict["processing_level"] = (
        "OK: files are already in Surya's format" if sdo and set(states) <= {"surya_netcdf", "aia_level_1_5"}
        else f"WARN: conversion needed, {needs} (AIA level 1 needs registration and exposure normalization; HMI needs rotating and rescaling to 0.6 arcsec)"
        if needs else "WARN: processing state unclear, see per-file states"
    )
    vector = any("vector" in n for f in files for n in f.notes)
    have_vec = any(c in {"hmi_bx", "hmi_by", "hmi_bz"} for i in instants for c in i.channels)
    verdict["hmi_vector_bx_by_bz"] = (
        "OK: Bx, By and Bz present" if have_vec
        else "WARN: vector field products seen, they must be converted to Bx, By, Bz" if vector
        else "WARN: no Bx, By or Bz; the channels would have to be filled in or the delivery completed"
    )
    broken = [f.path for f in files if any(n.startswith("could not read") or n == "no 2D image HDU" for n in f.notes)]
    if broken:
        verdict["unreadable_files"] = f"FAIL: {len(broken)} file(s) could not be read, e.g. {broken[0]}"
    others = [f for f in files if f.product and f.product not in ("aia", "hmi")]
    if others:
        verdict["other_instruments"] = f"WARN: {len(others)} file(s) from instruments Surya does not use, they will be ignored"
    captions = sum(m.n_nonempty_captions or 0 for m in metadata)
    verdict["captions"] = (
        f"OK: {captions} non-empty caption(s) found" if captions
        else "FAIL: no caption field found in the metadata files" if metadata else "FAIL: no metadata files found"
    )
    regions = sorted({r for m in metadata for r in m.region_fields})
    verdict["region_info"] = (
        f"OK: region-like fields {regions}" if regions
        else "WARN: no region fields (bbox, active region, heliographic position); token cropping needs one"
    )
    linked = sum(m.fits_linked for m in metadata)
    verdict["fits_linked_to_metadata"] = (
        f"OK: metadata mentions {linked} FITS name(s)" if linked
        else "WARN: no metadata file mentions a delivered FITS name, so the link between them is unclear"
    )
    return verdict


def inspect_delivery(root: Path, instant_tol_min: float = 6.0, pair_gap_min: float = 60.0, pair_tol_min: float = 6.0) -> DeliveryReport:
    """Scan a delivery folder and build the full report.

    Args:
        root: Folder to scan recursively.
        instant_tol_min: Files within this many minutes count as one instant.
        pair_gap_min: Wanted separation between the two input instants.
        pair_tol_min: Allowed deviation from pair_gap_min.

    Returns:
        The DeliveryReport.
    """
    paths = sorted(p for p in root.rglob("*") if p.is_file() and ".cache" not in p.parts)
    files: list[FileInfo] = []
    meta_paths: list[Path] = []
    counts: dict[str, int] = {"fits": 0, "netcdf": 0, "metadata": 0, "other": 0}
    for path in paths:
        suffix = path.suffix.lower()
        if suffix in FITS_SUFFIXES:
            counts["fits"] += 1
            files.append(inspect_fits(path, root))
        elif suffix in (".nc", ".nc4"):
            counts["netcdf"] += 1
            files.append(inspect_netcdf(path, root))
        elif suffix in METADATA_SUFFIXES:
            counts["metadata"] += 1
            meta_paths.append(path)
        else:
            counts["other"] += 1
    fits_names = [Path(f.path).name for f in files if f.kind == "fits"]
    metadata = [summarize_metadata(p, root, fits_names) for p in meta_paths]
    instants = group_instants(files, instant_tol_min)
    pairs, complete = count_pairs(instants, pair_gap_min, pair_tol_min)
    return DeliveryReport(
        root=str(root), counts=counts, files=files, metadata=metadata, instants=instants,
        pairs_60min=pairs, complete_pairs_60min=complete, verdict=assess(files, metadata, instants, pairs, complete),
    )


def print_report(report: DeliveryReport) -> None:
    """Print a short human-readable summary of a report.

    Args:
        report: The report to print.
    """
    print(f"Delivery: {report.root}")
    print(f"Files: {report.counts}")
    states: dict[str, int] = {}
    for f in report.files:
        states[f.state] = states.get(f.state, 0) + 1
    print(f"Processing states: {states}")
    shapes = sorted({str(f.shape) for f in report.files})
    print(f"Image shapes: {shapes}")
    print(f"Instants: {len(report.instants)}; pairs ~60 min apart: {report.pairs_60min} "
          f"(complete: {report.complete_pairs_60min})")
    for instant in report.instants[:12]:
        print(f"  {instant.t:%Y-%m-%d %H:%M} files={instant.n_files:<3} channels={len(instant.channels)}/13"
              + (f" missing={instant.missing}" if 0 < len(instant.missing) <= 4 else ""))
    if len(report.instants) > 12:
        print(f"  ... {len(report.instants) - 12} more")
    for m in report.metadata:
        print(f"Metadata {m.path}: records={m.n_records} captions={m.caption_fields} "
              f"region={m.region_fields} fits_linked={m.fits_linked}")
    notes = sorted({n for f in report.files for n in f.notes})
    for note in notes[:8]:
        print(f"Note: {note}")
    print("Verdict:")
    for name, text in report.verdict.items():
        print(f"  {name:<24} {text}")


def main() -> None:
    """Parse arguments, inspect the delivery and print or save the report."""
    parser = argparse.ArgumentParser(description="Inspect a data delivery for Surya.")
    parser.add_argument("root", type=Path, help="Folder with the delivered files.")
    parser.add_argument("--json-out", type=Path, help="Also write the full report as JSON.")
    parser.add_argument("--instant-tol-min", type=float, default=6.0)
    parser.add_argument("--pair-gap-min", type=float, default=60.0)
    parser.add_argument("--pair-tol-min", type=float, default=6.0)
    args = parser.parse_args()
    report = inspect_delivery(args.root.resolve(), args.instant_tol_min, args.pair_gap_min, args.pair_tol_min)
    print_report(report)
    if args.json_out:
        args.json_out.write_text(report.model_dump_json(indent=2))
        print(f"Full report: {args.json_out}")


if __name__ == "__main__":
    main()
