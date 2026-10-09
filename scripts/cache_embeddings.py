#!/usr/bin/env python3
"""cache_embeddings.py — Run frozen Surya once per instant and cache each region's pooled tokens.

For every instant in an events CSV (see build_events.py) it reads the Surya-format files at t-60 min and
t, builds Surya's input exactly as its own dataset class does, runs the encoder, and for each region at
that instant crops the token grid to the region's box, pools it to 16x16 = 256 tokens and saves it as a
(256, 1280) fp16 tensor. A JSONL manifest with one caption per sample is written for cached_dataset.py.

The Surya dataset class is not used because it also demands the file 60 minutes after t (its forecast
target), which this task does not need and which most instants of a local day do not have.

Usage:
  python scripts/cache_embeddings.py --events notebook/data/events_20140107.csv \\
      --roots notebook/data/Surya-1.0_validation_data --out data/cache --holdout 11946 11942
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import hdf5plugin  # noqa: F401  registers the compression filters the files use; must precede reading
import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "notebook"))
sys.path.insert(0, str(REPO / "scripts"))

from cached_dataset import ManifestRecord, write_manifest  # noqa: E402
from captions import RegionState, build_caption  # noqa: E402
from notebook_utils._paths import DATA_DIR  # noqa: E402
from notebook_utils.model import SuryaModel  # noqa: E402
from surya.datasets.helio import transform  # noqa: E402
from surya.utils.data import build_scalers  # noqa: E402
from tokens import crop_tokens, pool_tokens  # noqa: E402

INPUT_GAP_MINUTES = 60
TIME_DELTA_INPUT = [[1.0, 0.0]]  # (t - earlier) and (t - t) in hours, as Surya's dataset class returns


def find_file(instant: datetime, roots: list[Path]) -> Path:
    """Locate the Surya-format file for a grid instant under any of the given folders.

    Args:
        instant: A time on the 12-minute grid.
        roots: Folders to search, either flat or laid out as YYYY/MM/.

    Returns:
        The path of the file.

    Raises:
        FileNotFoundError: If no folder holds it.
    """
    name = f"{instant:%Y%m%d_%H%M}.nc"
    for root in roots:
        for candidate in (root / name, root / f"{instant:%Y/%m}" / name):
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"{name} not found under {[str(r) for r in roots]}")


class SuryaInput:
    """Reads and normalizes Surya-format files the way Surya's own dataset class does."""

    def __init__(self, config: dict, scalers_path: Path) -> None:
        """Load the channel order and the per-channel normalization constants.

        Args:
            config: Parsed Surya config.yaml.
            scalers_path: Path to Surya's scalers.yaml.
        """
        self.channels = config["data"]["sdo_channels"]
        scalers = build_scalers(info=yaml.safe_load(scalers_path.read_text()))
        self.means = np.array([scalers[c].mean for c in self.channels])
        self.stds = np.array([scalers[c].std for c in self.channels])
        self.epsilons = np.array([scalers[c].epsilon for c in self.channels])
        self.scale_factors = np.array([scalers[c].sl_scale_factor for c in self.channels])

    def load_normalized(self, path: Path) -> np.ndarray:
        """Read one file and apply Surya's signed-log normalization.

        Args:
            path: A Surya-format netCDF file.

        Returns:
            Array of shape (13, 4096, 4096), float32, channels in the config's order.
        """
        with xr.open_dataset(path, engine="h5netcdf", chunks=None, cache=False) as ds:
            data = ds[self.channels].to_array().load().to_numpy()
        return transform(data, self.means, self.stds, self.scale_factors, self.epsilons).astype(np.float32)

    def load_pair(self, instant: datetime, roots: list[Path]) -> dict[str, torch.Tensor]:
        """Build Surya's input for an instant: the files 60 minutes before and at the instant.

        Args:
            instant: Reference time t on the 12-minute grid.
            roots: Folders to search for the files.

        Returns:
            A batch dict with `ts` of shape (1, 13, 2, 4096, 4096) and `time_delta_input` of shape (1, 2).
        """
        earlier = instant - timedelta(minutes=INPUT_GAP_MINUTES)
        frames = [self.load_normalized(find_file(t, roots)) for t in (earlier, instant)]
        ts = torch.from_numpy(np.stack(frames, axis=1)).unsqueeze(0)
        return {"ts": ts, "time_delta_input": torch.tensor(TIME_DELTA_INPUT)}


def region_state(row: pd.Series) -> RegionState:
    """Turn one events row into the fields the caption needs.

    Args:
        row: A row of the events CSV.

    Returns:
        The region state, with missing catalog values as None.
    """
    def opt(value):
        return None if pd.isna(value) else value

    return RegionState(
        latitude=row.latitude, longitude=row.longitude, radial_fraction=row.radial_fraction,
        n_spots=None if pd.isna(row.n_spots) else int(row.n_spots),
        mcintosh=opt(row.mcintosh), magnetic=opt(row.magnetic), flare_class=opt(row.flare_class),
    )


def main() -> None:
    """Embed every instant once, cache each region's pooled tokens, and write the manifests."""
    parser = argparse.ArgumentParser(description="Cache pooled Surya tokens for each region sample.")
    parser.add_argument("--events", type=Path, required=True, nargs="+", help="CSV(s) from build_events.py")
    parser.add_argument("--roots", type=Path, required=True, nargs="+", help="Folders holding the .nc files")
    parser.add_argument("--out", type=Path, default=Path("data/cache"))
    parser.add_argument("--holdout", type=int, nargs="*", default=[], help="NOAA numbers kept out of training")
    parser.add_argument("--limit-instants", type=int, default=None, help="Only process this many instants (testing)")
    args = parser.parse_args()

    events = pd.concat([pd.read_csv(p, parse_dates=["instant"]) for p in args.events], ignore_index=True)
    instants = sorted(events.instant.unique())
    if args.limit_instants:
        instants = instants[: args.limit_instants]
        events = events[events.instant.isin(instants)]
    (args.out / "emb").mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load((DATA_DIR / "Surya-1.0" / "config.yaml").read_text())
    reader = SuryaInput(config, DATA_DIR / "Surya-1.0" / "scalers.yaml")
    surya = SuryaModel.load(config=config)
    print(f"{len(events)} samples over {len(instants)} instants; holdout regions {args.holdout}", flush=True)

    train, heldout = [], []
    for k, instant in enumerate(instants, start=1):
        t = pd.Timestamp(instant).to_pydatetime()
        rows = events[events.instant == instant]
        tokens = surya.embed_batch(reader.load_pair(t, args.roots))  # (1, 65536, 1280)
        for _, row in rows.iterrows():
            crop = crop_tokens(tokens, (row.x_min, row.y_min, row.x_max, row.y_max))
            pooled = pool_tokens(crop).squeeze(0).to(torch.float16).cpu()  # (256, 1280)
            relative = f"emb/{row.sample_id}.pt"
            torch.save(pooled, args.out / relative)
            record = ManifestRecord(id=row.sample_id, embedding_path=relative, caption=build_caption(region_state(row)))
            (heldout if int(row.noaa) in args.holdout else train).append(record)
        print(f"[{k}/{len(instants)}] {t:%H:%M}: {len(rows)} samples, crop grid {tuple(crop.shape[1:3])}", flush=True)
        del tokens
        torch.cuda.empty_cache()

    write_manifest(train, args.out / "manifest.jsonl")
    if heldout:
        write_manifest(heldout, args.out / "manifest_heldout.jsonl")
    print(f"train {len(train)} -> {args.out / 'manifest.jsonl'}; held out {len(heldout)}", flush=True)


if __name__ == "__main__":
    main()
