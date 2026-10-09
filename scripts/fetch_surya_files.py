#!/usr/bin/env python3
"""fetch_surya_files.py — Download Surya-format SDO files for given instants from the public S3 bucket.

For each instant t it fetches the file at t and the one 60 minutes earlier (what Surya takes as input).
Files are `s3://nasa-surya-bench/YYYY/MM/YYYYMMDD_HHMM.nc`, one every 12 minutes, about 585 MB each.
From the cluster a single connection runs at about 0.26 MB/s, so each file is split into chunks fetched
by several threads at once, and the download can be stopped and restarted without losing finished chunks.

Usage:
  python scripts/fetch_surya_files.py --times-file configs/paper_event_times.txt
  python scripts/fetch_surya_files.py --times 2012-07-05T03:36 --workers 24
"""

import argparse
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

BASE_URL = "https://nasa-surya-bench.s3.amazonaws.com/"
GRID_MINUTES = 12
CHUNK_BYTES = 8 * 1024 * 1024
REQUEST_TIMEOUT_S = 60
MAX_RETRIES = 6


def snap_to_grid(instant: datetime) -> datetime:
    """Round an instant to the nearest point of the 12-minute grid that the bucket uses.

    Args:
        instant: Any time.

    Returns:
        The nearest time whose minute-of-day is a multiple of 12.
    """
    minutes = round((instant.hour * 60 + instant.minute + instant.second / 60) / GRID_MINUTES) * GRID_MINUTES
    return datetime(instant.year, instant.month, instant.day) + timedelta(minutes=minutes)


def object_key(instant: datetime) -> str:
    """Name of the file for a grid instant inside the bucket.

    Args:
        instant: A time on the 12-minute grid.

    Returns:
        The key, for example '2012/07/20120705_0336.nc'.
    """
    return f"{instant:%Y/%m}/{instant:%Y%m%d_%H%M}.nc"


def files_needed(instants: list[datetime], with_previous: bool = True) -> list[datetime]:
    """Grid instants whose files are needed, in order and without repeats.

    Args:
        instants: Times of interest; each is snapped to the grid.
        with_previous: Also include the instant 60 minutes before each one.

    Returns:
        Grid instants, the earlier file of each pair first.
    """
    seen: dict[datetime, None] = {}
    for instant in instants:
        t = snap_to_grid(instant)
        if with_previous:
            seen.setdefault(t - timedelta(minutes=60), None)
        seen.setdefault(t, None)
    return list(seen)


def remote_size(key: str) -> int | None:
    """Size in bytes of an object in the bucket, from a HEAD request.

    Args:
        key: Object key.

    Returns:
        The size, or None if the object does not exist.
    """
    request = urllib.request.Request(BASE_URL + key, method="HEAD")
    try:
        return int(urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S).headers["Content-Length"])
    except urllib.error.HTTPError as error:
        if error.code in (403, 404):
            return None
        raise


class Progress:
    """Thread-safe byte counter, so the aggregate speed can be reported."""

    def __init__(self, total_bytes: int) -> None:
        self.total = total_bytes
        self.done = 0
        self.lock = threading.Lock()
        self.started = time.time()

    def add(self, n: int) -> None:
        with self.lock:
            self.done += n

    def line(self) -> str:
        elapsed = max(time.time() - self.started, 1e-6)
        speed = self.done / elapsed / 1e6
        left = (self.total - self.done) / max(self.done / elapsed, 1)
        return f"{self.done / 1e6:8.0f} / {self.total / 1e6:.0f} MB  {speed:5.2f} MB/s  ~{left / 60:.0f} min left"


def fetch_chunk(key: str, fd: int, start: int, end: int, progress: Progress) -> int:
    """Download one byte range and write it at its offset in the open file.

    Args:
        key: Object key.
        fd: File descriptor of the preallocated partial file.
        start: First byte, inclusive.
        end: Last byte, inclusive.
        progress: Counter to update.

    Returns:
        The start offset, to mark the chunk as finished.

    Raises:
        RuntimeError: If the chunk still fails after all retries.
    """
    for attempt in range(MAX_RETRIES):
        try:
            request = urllib.request.Request(BASE_URL + key, headers={"Range": f"bytes={start}-{end}"})
            data = urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S).read()
            if len(data) != end - start + 1:
                raise OSError(f"short read: {len(data)} of {end - start + 1} bytes")
            os.pwrite(fd, data, start)
            progress.add(len(data))
            return start
        except (OSError, urllib.error.URLError) as error:
            time.sleep(min(2**attempt, 30))
            last = error
    raise RuntimeError(f"{key} bytes {start}-{end} failed after {MAX_RETRIES} tries: {last}")


def download_file(key: str, size: int, dest: Path, workers: int, progress: Progress) -> Path:
    """Download one file with parallel range requests, resuming from finished chunks.

    A sidecar `<name>.done` lists the start offset of every chunk already written, so an interrupted
    run continues instead of starting over. The file only gets its final name once it is complete.

    Args:
        key: Object key.
        size: Expected size in bytes.
        dest: Folder that mirrors the bucket's layout.
        workers: Simultaneous connections.
        progress: Counter shared across files.

    Returns:
        The final path.
    """
    final = dest / key
    final.parent.mkdir(parents=True, exist_ok=True)
    part = final.with_suffix(".nc.part")
    done_file = final.with_suffix(".nc.done")
    done = set(int(x) for x in done_file.read_text().split()) if done_file.exists() and part.exists() else set()
    fd = os.open(part, os.O_RDWR | os.O_CREAT)
    try:
        if os.fstat(fd).st_size != size:
            os.ftruncate(fd, size)
            done.clear()
        starts = [s for s in range(0, size, CHUNK_BYTES) if s not in done]
        progress.add(sum(min(CHUNK_BYTES, size - s) for s in done))
        with ThreadPoolExecutor(max_workers=workers) as pool, open(done_file, "a") as log:
            futures = [pool.submit(fetch_chunk, key, fd, s, min(s + CHUNK_BYTES, size) - 1, progress) for s in starts]
            for future in as_completed(futures):
                log.write(f"{future.result()}\n")
                log.flush()
    finally:
        os.close(fd)
    if part.stat().st_size != size:
        raise RuntimeError(f"{part}: wrong size after download")
    part.rename(final)
    done_file.unlink(missing_ok=True)
    return final


def read_times(path: Path) -> list[datetime]:
    """Read instants from a text file, one ISO time per line; '#' starts a comment.

    Args:
        path: The file.

    Returns:
        The instants in file order.
    """
    out = []
    for line in path.read_text().splitlines():
        text = line.split("#", 1)[0].strip()
        if text:
            out.append(datetime.fromisoformat(text))
    return out


def main() -> None:
    """Parse arguments, check what is already local, and download the rest in order."""
    parser = argparse.ArgumentParser(description="Fetch Surya-format SDO files for given instants.")
    parser.add_argument("--times", nargs="*", default=[], help="ISO instants, e.g. 2012-07-05T03:36")
    parser.add_argument("--times-file", type=Path, help="Text file with one ISO instant per line.")
    parser.add_argument("--dest", type=Path, default=Path("notebook/data/surya_events"))
    parser.add_argument("--workers", type=int, default=16, help="Simultaneous connections per file.")
    parser.add_argument("--no-previous", action="store_true", help="Skip the file 60 minutes earlier.")
    args = parser.parse_args()

    instants = [datetime.fromisoformat(t) for t in args.times]
    if args.times_file:
        instants += read_times(args.times_file)
    if not instants:
        sys.exit("no instants given (--times or --times-file)")

    wanted = files_needed(instants, with_previous=not args.no_previous)
    plan = []
    for t in wanted:
        key = object_key(t)
        size = remote_size(key)
        if size is None:
            print(f"MISSING in bucket: {key}", flush=True)
            continue
        have = (args.dest / key).exists() and (args.dest / key).stat().st_size == size
        plan.append((key, size, have))
        print(f"{key}: {size / 1e6:.0f} MB {'(already complete)' if have else ''}", flush=True)
    todo = [(k, s) for k, s, have in plan if not have]
    if not todo:
        print("Nothing to download.")
        return

    progress = Progress(sum(s for _, s in todo))
    stop = threading.Event()

    def report() -> None:
        while not stop.wait(30):
            print(progress.line(), flush=True)

    threading.Thread(target=report, daemon=True).start()
    for key, size in todo:
        print(f"-> {key}", flush=True)
        download_file(key, size, args.dest, args.workers, progress)
        print(f"   complete: {progress.line()}", flush=True)
    stop.set()
    print("All files complete.")


if __name__ == "__main__":
    main()
