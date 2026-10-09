#!/usr/bin/env python3
"""build_paper_events.py — Hand-placed samples for the prominence eruptions of Labrosse et al. (2012).

These limb events have no active region in the catalog, so there is nothing to look up. Each box was
placed by eye on AIA 304 images of the instant (2026-10-09), starting from the HEK eruption position, and
each caption restates what the paper's own text says about that moment. The result is the same CSV
layout as build_events.py plus a `caption` column that cache_embeddings.py uses as written.

The Song et al. (2018) event is not here: it is an active region, so build_events.py covers it.

Usage:
  python scripts/build_paper_events.py --out notebook/data/events_labrosse2012.csv
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_events import box_around  # noqa: E402

HALF_SIZE = 352

# (label, instant, box center column, row, caption). Centers were chosen from AIA 304 views.
EVENTS = [
    # Fig. 2, 2010-06-13, north-west limb: "from around 03:00 the top part begins to detach", "at around
    # 05:00 there is a clear detachment of the structure and this then moves outward"; tracking from 04:15.
    ("fig2_0412", datetime(2010, 6, 13, 4, 12), 3500, 2850,
     "A prominence eruption above the north-west limb: the top part of the prominence is starting to detach from its base."),
    ("fig2_0500", datetime(2010, 6, 13, 5, 0), 3500, 2850,
     "A prominence eruption above the north-west limb: the structure has clearly detached from its base and moves outward."),
    # Fig. 3, 2010-09-08, north-west limb: "appreciable outward motion not until approximately 23:08".
    ("fig3_2312", datetime(2010, 9, 8, 23, 12), 3560, 2850,
     "A prominence eruption above the north-west limb, just after its outward motion has begun."),
    ("fig3_2324", datetime(2010, 9, 8, 23, 24), 3560, 2850,
     "A prominence eruption above the north-west limb, with plasma streaming outward away from the Sun."),
    # Fig. 4, 2011-03-19, south-west: "a large kinking loop eruption ... helical motions are visible in the
    # prominence legs".
    ("fig4_1148", datetime(2011, 3, 19, 11, 48), 3760, 1180,
     "A large kinking loop prominence eruption at the south-west limb, with helical motions in the legs of the prominence."),
    ("fig4_1200", datetime(2011, 3, 19, 12, 0), 3760, 1180,
     "A large kinking loop prominence eruption at the south-west limb, with helical motions in the legs of the prominence."),
    # Fig. 5, 2011-06-10, south-west limb: "first appearing as a compact flaring region ... soon developed
    # into a very tight looped structure at around 17:49" (the 18:00 file does not exist in the bucket).
    ("fig5_1748", datetime(2011, 6, 10, 17, 48), 3490, 1190,
     "A compact bright flaring region at the south-west limb that is developing into a tight looped prominence structure."),
]


def build_rows() -> pd.DataFrame:
    """Turn the hand-placed events into the events CSV layout.

    Returns:
        One row per event with the box edges clamped inside the image, `noaa` set to 0 because there is
        no region, and the caption to use as written.
    """
    rows = []
    for label, instant, cx, cy, caption in EVENTS:
        x_min, y_min, x_max, y_max = box_around(cx, cy, HALF_SIZE)
        rows.append({
            "sample_id": f"{instant:%Y%m%d_%H%M}_paper_{label}", "instant": instant, "noaa": 0,
            "x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max, "caption": caption,
        })
    return pd.DataFrame(rows)


def main() -> None:
    """Write the events CSV."""
    parser = argparse.ArgumentParser(description="Write the hand-placed Labrosse 2012 events.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    frame = build_rows()
    frame.to_csv(args.out, index=False)
    print(f"{len(frame)} samples -> {args.out}")


if __name__ == "__main__":
    main()
