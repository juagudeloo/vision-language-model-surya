"""captions.py — Build a caption for an active region from its NOAA catalog entry.

Only facts that can be seen in the image go into the text: where the region sits on the disk, how many
sunspots it has, its McIntosh and Mount Wilson classes, and whether a flare is in progress. The NOAA region
number is left out on purpose: it cannot be read from pixels, and with few samples the model would only
memorize it.

Class definitions follow the Royal Observatory of Belgium's SIDC page
https://sidc.be/educational/classification.php (checked 2026-10-09).
"""

from pydantic import BaseModel, Field

ZURICH = {
    "A": "a small single unipolar sunspot",
    "B": "a bipolar group with no penumbra",
    "C": "a bipolar group where one spot has penumbra",
    "D": "a bipolar group with penumbra at both ends and an extent of at most 10 degrees",
    "E": "a bipolar group with penumbra at both ends and an extent of 10 to 15 degrees",
    "F": "an elongated bipolar group with penumbra at both ends and an extent above 15 degrees",
    "H": "a unipolar group with penumbra",
}
PENUMBRA = {
    "x": None,
    "r": "a rudimentary penumbra",
    "s": "a small symmetric penumbra",
    "a": "a small asymmetric penumbra",
    "h": "a large symmetric penumbra",
    "k": "a large asymmetric penumbra",
}
COMPACTNESS = {
    "x": None,
    "o": "few spots lie between its leading and following ends",
    "i": "numerous spots lie between its ends, none with mature penumbra",
    "c": "its interior is compact, with many strong spots",
}
# The catalog also returns corrupted strings such as 'ALPHAGAMMA-DELTA' for a single spot; only these
# canonical values are translated, anything else is left out of the caption.
MAGNETIC = {
    "ALPHA": "a unipolar magnetic configuration",
    "BETA": "a bipolar magnetic configuration with a simple division between the polarities",
    "BETA-GAMMA": "a bipolar magnetic configuration with no continuous line separating the polarities",
    "GAMMA": "a complex magnetic configuration, too irregular to call bipolar",
    "BETA-DELTA": "a bipolar magnetic configuration with opposite-polarity umbrae in one penumbra",
    "BETA-GAMMA-DELTA": "a complex magnetic configuration with opposite-polarity umbrae in one penumbra",
}


class RegionState(BaseModel):
    """One active region at one instant, as needed to write its caption."""

    latitude: float = Field(description="Heliographic latitude in degrees, north positive.")
    longitude: float = Field(description="Stonyhurst longitude in degrees, west positive.")
    radial_fraction: float = Field(description="Distance from disk center as a fraction of the solar radius.")
    n_spots: int | None = Field(default=None, description="Sunspot count from the catalog, if given.")
    mcintosh: str | None = Field(default=None, description="Three-letter McIntosh class, if given.")
    magnetic: str | None = Field(default=None, description="Mount Wilson class as the catalog writes it, if given.")
    flare_class: str | None = Field(default=None, description="GOES class of a flare in progress in this region.")


def describe_mcintosh(code: str | None) -> str | None:
    """Translate a McIntosh class into words, or None if it is not a valid class.

    Args:
        code: Three letters such as 'FKC'.

    Returns:
        A phrase such as 'an elongated bipolar group ...; its largest spot has a large asymmetric
        penumbra and its interior is compact, ...', or None when the code is missing or any letter is
        not valid for its position.
    """
    if not code or len(code) != 3:
        return None
    z, p, c = code[0].upper(), code[1].lower(), code[2].lower()
    if z not in ZURICH or p not in PENUMBRA or c not in COMPACTNESS:
        return None
    extras = []
    if PENUMBRA[p]:
        extras.append(f"its largest spot has {PENUMBRA[p]}")
    if COMPACTNESS[c]:
        extras.append(COMPACTNESS[c])
    return ZURICH[z] + ("; " + " and ".join(extras) if extras else "")


def describe_magnetic(label: str | None) -> str | None:
    """Translate a Mount Wilson class into words, or None if it is not a canonical value.

    Args:
        label: The catalog string, for example 'BETA-GAMMA-DELTA'.

    Returns:
        A phrase, or None for missing or corrupted strings.
    """
    return MAGNETIC.get((label or "").strip().upper())


def describe_location(latitude: float, longitude: float, radial_fraction: float) -> str:
    """Say where on the disk a region is, in words plus its heliographic coordinates.

    Args:
        latitude: Degrees, north positive.
        longitude: Degrees, west positive.
        radial_fraction: Distance from disk center in solar radii.

    Returns:
        For example 'in the southern hemisphere near the central meridian (S09 W11)'.
    """
    hemisphere = "northern" if latitude >= 0 else "southern"
    side = "west" if longitude >= 0 else "east"
    if radial_fraction > 0.85:
        where = f"close to the {side} limb"
    elif abs(longitude) < 15:
        where = "near the central meridian"
    else:
        where = f"on the {side} side of the disk"
    coords = f"{'N' if latitude >= 0 else 'S'}{abs(latitude):02.0f} {'W' if longitude >= 0 else 'E'}{abs(longitude):02.0f}"
    return f"in the {hemisphere} hemisphere {where} ({coords})"


def build_caption(state: RegionState) -> str:
    """Write the caption for one region at one instant.

    Args:
        state: The region's catalog and position data.

    Returns:
        Two or three sentences: where it is and how many spots it has, what its group and magnetic
        classes say, and the flare in progress if there is one.
    """
    spots = ""
    if state.n_spots is not None:
        spots = f" with {state.n_spots} sunspot{'s' if state.n_spots != 1 else ''}"
    sentences = [f"An active region {describe_location(state.latitude, state.longitude, state.radial_fraction)}{spots}."]
    group = describe_mcintosh(state.mcintosh)
    magnetic = describe_magnetic(state.magnetic)
    if group:
        sentences.append(f"It is {group}.")
    if magnetic:
        sentences.append(f"It shows {magnetic}.")
    if state.flare_class:
        article = "An" if state.flare_class[:1].upper() in "AMX" else "A"
        sentences.append(f"{article} {state.flare_class} flare is in progress.")
    return " ".join(sentences)
