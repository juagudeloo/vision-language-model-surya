#!/usr/bin/env python3
"""build_events.py — List (active region, instant) samples for a day, with position and caption fields.

Active regions come from the NOAA SWPC Observer entries in HEK, which carry a position, sunspot count,
McIntosh class and Mount Wilson class once per day (at 00:00 UT). Each position is carried to every
requested instant with differential rotation, converted to pixels of Surya's 4096x4096 grid, and turned
into a square box. Flares in progress come from the SWPC flare list in HEK.

Usage:
  python scripts/build_events.py --day 2014-01-07 --first 16:00 --last 19:24 --out notebook/data/events_20140107.csv
"""

import argparse
import math
from datetime import datetime, timedelta
from pathlib import Path

import astropy.units as u
import pandas as pd
from astropy.coordinates import SkyCoord
from pydantic import BaseModel, Field
from sunpy.coordinates import Helioprojective, HeliographicStonyhurst, sun
from sunpy.net import attrs as a
from sunpy.net import hek
from sunpy.physics.differential_rotation import solar_rotate_coordinate

IMAGE_SIZE = 4096
ARCSEC_PER_PIXEL = 0.6
CENTER_PX = 2047.5
GRID_MINUTES = 12
MAX_RADIAL_FRACTION = 0.92
CATALOG_PROVIDER = "NOAA SWPC Observer"


class ActiveRegion(BaseModel):
    """A NOAA active region as listed once per day in the catalog."""

    noaa: int = Field(description="NOAA region number.")
    catalog_time: datetime = Field(description="Time the catalog position refers to.")
    x_arcsec: float = Field(description="Helioprojective x at catalog_time, west positive.")
    y_arcsec: float = Field(description="Helioprojective y at catalog_time, north positive.")
    n_spots: int | None = Field(default=None, description="Sunspot count.")
    mcintosh: str | None = Field(default=None, description="McIntosh class.")
    magnetic: str | None = Field(default=None, description="Mount Wilson class as the catalog writes it.")


class Flare(BaseModel):
    """A flare from the SWPC list."""

    start: datetime
    peak: datetime
    end: datetime
    goes_class: str
    noaa: int | None = Field(default=None, description="Region that produced it, if the catalog says.")


class EventRow(BaseModel):
    """One sample: a region at an instant, with its box in pixels and the fields for its caption."""

    sample_id: str = Field(description="Instant and region number, e.g. 20140107_1824_AR11944.")
    instant: datetime = Field(description="Reference time t; Surya also reads the file 60 minutes earlier.")
    noaa: int = Field(description="NOAA region number; kept as metadata, not written in the caption.")
    x_arcsec: float = Field(description="Position at the instant, west positive.")
    y_arcsec: float = Field(description="Position at the instant, north positive.")
    latitude: float = Field(description="Heliographic latitude, north positive.")
    longitude: float = Field(description="Stonyhurst longitude, west positive.")
    radial_fraction: float = Field(description="Distance from disk center in solar radii.")
    px: float = Field(description="Column of the position in the 4096x4096 image.")
    py: float = Field(description="Row of the position in the 4096x4096 image (before any flip).")
    x_min: int = Field(description="Left edge of the box in pixels.")
    y_min: int = Field(description="Top edge of the box in pixels.")
    x_max: int = Field(description="Right edge of the box in pixels.")
    y_max: int = Field(description="Bottom edge of the box in pixels.")
    n_spots: int | None = None
    mcintosh: str | None = None
    magnetic: str | None = None
    flare_class: str | None = Field(default=None, description="GOES class of a flare in progress in this region.")


def _clean(value) -> str | None:
    text = str(value).strip()
    return None if text in ("", "--", "masked", "None") else text


def fetch_regions(day: datetime) -> list[ActiveRegion]:
    """Read the NOAA SWPC active-region list for a day from HEK.

    Args:
        day: Midnight of the day.

    Returns:
        One entry per region, using the catalog's 00:00 UT position.
    """
    table = hek.HEKClient().search(
        a.Time(day, day + timedelta(hours=1)), a.hek.EventType("AR"), a.hek.FRM.Name == CATALOG_PROVIDER
    )
    regions: dict[int, ActiveRegion] = {}
    for row in table:
        number = _clean(row["ar_noaanum"])
        if number is None or int(number) in regions:
            continue
        spots = _clean(row["ar_numspots"])
        regions[int(number)] = ActiveRegion(
            noaa=int(number),
            catalog_time=datetime.fromisoformat(str(row["event_starttime"])),
            x_arcsec=float(row["hpc_x"]),
            y_arcsec=float(row["hpc_y"]),
            n_spots=int(float(spots)) if spots else None,
            mcintosh=_clean(row["ar_mcintoshcls"]),
            magnetic=_clean(row["ar_mtwilsoncls"]),
        )
    return list(regions.values())


def fetch_flares(start: datetime, end: datetime) -> list[Flare]:
    """Read the SWPC flare list between two times from HEK.

    Args:
        start: Window start.
        end: Window end.

    Returns:
        Flares, without the duplicate rows other providers add.
    """
    table = hek.HEKClient().search(a.Time(start, end), a.hek.EventType("FL"), a.hek.FRM.Name == "SWPC")
    flares: dict[tuple, Flare] = {}
    for row in table:
        number = _clean(row["ar_noaanum"])
        flare = Flare(
            start=datetime.fromisoformat(str(row["event_starttime"])),
            peak=datetime.fromisoformat(str(row["event_peaktime"])),
            end=datetime.fromisoformat(str(row["event_endtime"])),
            goes_class=str(row["fl_goescls"]),
            noaa=int(number) if number and number != "0" else None,
        )
        flares[(flare.start, flare.goes_class, flare.noaa)] = flare
    return list(flares.values())


def rotate_region(region: ActiveRegion, instant: datetime) -> tuple[float, float, float, float, float] | None:
    """Carry a catalog position to another instant with differential rotation.

    Args:
        region: The region and its catalog position.
        instant: The time to move it to.

    Returns:
        (x_arcsec, y_arcsec, latitude, longitude, radial_fraction) at the instant, or None if the
        region is behind the limb or too close to it to be useful.
    """
    start = SkyCoord(
        region.x_arcsec * u.arcsec, region.y_arcsec * u.arcsec,
        frame=Helioprojective(obstime=region.catalog_time, observer="earth"),
    )
    moved = solar_rotate_coordinate(start, time=instant)
    x, y = moved.Tx.to_value(u.arcsec), moved.Ty.to_value(u.arcsec)
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    radial = math.hypot(x, y) / sun.angular_radius(instant).to_value(u.arcsec)
    if radial > MAX_RADIAL_FRACTION:
        return None
    stonyhurst = moved.transform_to(HeliographicStonyhurst(obstime=instant))
    return x, y, stonyhurst.lat.to_value(u.deg), stonyhurst.lon.to_value(u.deg), radial


def arcsec_to_pixel(x_arcsec: float, y_arcsec: float) -> tuple[float, float]:
    """Convert helioprojective arcseconds to (column, row) in Surya's 4096x4096 image.

    Assumes disk center at pixel 2047.5 (0-based) and 0.6 arcsec per pixel, as in the level-1.5 headers,
    with rows increasing northward. Checked against the data on 2026-10-09: on 2014-01-07 18:24 the HMI
    field strength (smoothed) at the predicted positions of six regions summed to 2198 G with rows north
    and to 225 G with the rows flipped, and the two strongest polarity concentrations of AR 11944 lay on
    either side of its predicted position.

    Args:
        x_arcsec: West positive.
        y_arcsec: North positive.

    Returns:
        (column, row) as floats.
    """
    return CENTER_PX + x_arcsec / ARCSEC_PER_PIXEL, CENTER_PX + y_arcsec / ARCSEC_PER_PIXEL


def box_around(px: float, py: float, half_size: int) -> tuple[int, int, int, int]:
    """A square box centered on a pixel, moved inward if it would leave the image.

    Args:
        px: Column.
        py: Row.
        half_size: Half the side in pixels.

    Returns:
        (x_min, y_min, x_max, y_max) with x_max and y_max exclusive, both inside 0..4096.
    """
    side = 2 * half_size
    x_min = min(max(round(px) - half_size, 0), IMAGE_SIZE - side)
    y_min = min(max(round(py) - half_size, 0), IMAGE_SIZE - side)
    return x_min, y_min, x_min + side, y_min + side


def flare_in_progress(flares: list[Flare], noaa: int, instant: datetime) -> str | None:
    """GOES class of a flare from this region that is under way at the instant.

    Args:
        flares: Flares for the day.
        noaa: Region number.
        instant: Time to check.

    Returns:
        The class, e.g. 'X1.2', or None.
    """
    for flare in flares:
        if flare.noaa == noaa and flare.start <= instant <= flare.end:
            return flare.goes_class
    return None


def grid_instants(day: datetime, first: str, last: str) -> list[datetime]:
    """Instants on the 12-minute grid between two clock times of a day, inclusive.

    Args:
        day: Midnight of the day.
        first: 'HH:MM' of the first instant.
        last: 'HH:MM' of the last instant.

    Returns:
        The instants, each a multiple of 12 minutes after midnight.
    """
    t = day.replace(hour=int(first[:2]), minute=int(first[3:]))
    stop = day.replace(hour=int(last[:2]), minute=int(last[3:]))
    out = []
    while t <= stop:
        if (t.hour * 60 + t.minute) % GRID_MINUTES == 0:
            out.append(t)
        t += timedelta(minutes=GRID_MINUTES)
    return out


def build_events(day: datetime, instants: list[datetime], half_size: int = 352) -> list[EventRow]:
    """Build the sample list for a day.

    Args:
        day: Midnight of the day.
        instants: Reference times, on the 12-minute grid.
        half_size: Half the box side in pixels; 352 gives 704 px, 44x44 tokens, as in notebook 02.

    Returns:
        One row per usable (region, instant) pair, sorted by instant then region.
    """
    regions = fetch_regions(day)
    flares = fetch_flares(day, day + timedelta(days=1))
    rows = []
    for instant in instants:
        for region in regions:
            moved = rotate_region(region, instant)
            if moved is None:
                continue
            x, y, lat, lon, radial = moved
            px, py = arcsec_to_pixel(x, y)
            x_min, y_min, x_max, y_max = box_around(px, py, half_size)
            rows.append(EventRow(
                sample_id=f"{instant:%Y%m%d_%H%M}_AR{region.noaa}", instant=instant, noaa=region.noaa,
                x_arcsec=x, y_arcsec=y, latitude=lat, longitude=lon, radial_fraction=radial,
                px=px, py=py, x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max,
                n_spots=region.n_spots, mcintosh=region.mcintosh, magnetic=region.magnetic,
                flare_class=flare_in_progress(flares, region.noaa, instant),
            ))
    return rows


def main() -> None:
    """Parse arguments, build the sample list, and write it as CSV."""
    parser = argparse.ArgumentParser(description="List active-region samples for a day.")
    parser.add_argument("--day", required=True, help="YYYY-MM-DD")
    parser.add_argument("--first", required=True, help="HH:MM of the first instant (on the 12-minute grid)")
    parser.add_argument("--last", required=True, help="HH:MM of the last instant")
    parser.add_argument("--half-size", type=int, default=352)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    day = datetime.fromisoformat(args.day)
    rows = build_events(day, grid_instants(day, args.first, args.last), args.half_size)
    pd.DataFrame([r.model_dump() for r in rows]).to_csv(args.out, index=False)
    print(f"{len(rows)} samples, {len({r.noaa for r in rows})} regions, {len({r.instant for r in rows})} instants -> {args.out}")


if __name__ == "__main__":
    main()
