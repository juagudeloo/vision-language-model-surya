"""cached_dataset.py — Dataset over cached, pooled Surya embeddings and their captions.

Surya is frozen, so its output for an observation never changes. It is computed once, cropped to
the region of interest and pooled to 256 tokens (see notebook 02), saved to disk, and listed in a
JSONL manifest next to its caption. Training then never loads Surya.
"""

import json
from pathlib import Path

import torch
from pydantic import BaseModel, Field
from torch import Tensor
from torch.utils.data import Dataset

N_VISUAL_TOKENS = 256
SURYA_DIM = 1280


class ManifestRecord(BaseModel):
    """One training sample: where its cached embedding lives and the caption to learn."""

    id: str = Field(description="Unique name of the sample, used in logs and generated captions.")
    embedding_path: str = Field(
        description="Path to a saved (256, 1280) tensor, relative to the manifest's folder."
    )
    caption: str = Field(min_length=1, description="Target text the model should produce.")


def read_manifest(path: str | Path) -> list[ManifestRecord]:
    """Read a JSONL manifest, one JSON object per line.

    Args:
        path: The manifest file.

    Returns:
        The records in file order.

    Raises:
        ValueError: If a line is not valid, or two records share an id.
    """
    path = Path(path)
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(ManifestRecord.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(f"{path}:{number}: {exc}") from exc
    ids = [r.id for r in records]
    duplicated = sorted({i for i in ids if ids.count(i) > 1})
    if duplicated:
        raise ValueError(f"{path}: duplicate ids {duplicated}")
    return records


def write_manifest(records: list[ManifestRecord], path: str | Path) -> None:
    """Write records as a JSONL manifest.

    Args:
        records: The records to write.
        path: Destination file; its folder is where relative embedding paths resolve.
    """
    Path(path).write_text("".join(r.model_dump_json() + "\n" for r in records), encoding="utf-8")


class CachedEmbeddingDataset(Dataset):
    """Samples of (cached visual tokens, caption), checked up front so a bad manifest fails fast.

    Every file is checked for existence when the dataset is built, because on the cluster the
    expensive part is loading the language model, and a typo in a path should not cost that.
    Tensor shapes are checked on first read.
    """

    def __init__(self, manifest_path: str | Path) -> None:
        """Read the manifest and check that every embedding file exists.

        Args:
            manifest_path: JSONL manifest; relative embedding paths resolve from its folder.

        Raises:
            ValueError: If the manifest is empty or malformed.
            FileNotFoundError: If any embedding file is missing, naming all of them.
        """
        self.manifest_path = Path(manifest_path)
        self.records = read_manifest(self.manifest_path)
        if not self.records:
            raise ValueError(f"{self.manifest_path}: no records")
        root = self.manifest_path.parent
        self.paths = [root / r.embedding_path for r in self.records]
        missing = [str(p) for p in self.paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} embedding file(s) missing, e.g. {missing[:3]}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        """Load one sample.

        Args:
            index: Position in the manifest.

        Returns:
            A dict with `id` (str), `visual` (Tensor of shape (256, 1280), the dtype it was saved
            in, normally float16) and `caption` (str).

        Raises:
            ValueError: If the saved tensor does not have shape (256, 1280).
        """
        record = self.records[index]
        visual = torch.load(self.paths[index], map_location="cpu", weights_only=True)
        if tuple(visual.shape) != (N_VISUAL_TOKENS, SURYA_DIM):
            raise ValueError(
                f"{self.paths[index]}: expected shape {(N_VISUAL_TOKENS, SURYA_DIM)}, got {tuple(visual.shape)}"
            )
        return {"id": record.id, "visual": visual, "caption": record.caption}


def collate_samples(samples: list[dict]) -> dict:
    """Stack a list of samples into a batch, leaving the captions as text.

    Tokenizing the captions is the training step's job, because how the text is laid out around
    the image tokens is a modeling decision, not a data one.

    Args:
        samples: Items returned by CachedEmbeddingDataset.

    Returns:
        A dict with `ids` (list[str]), `visual` (Tensor of shape (B, 256, 1280)) and
        `captions` (list[str]).
    """
    return {
        "ids": [s["id"] for s in samples],
        "visual": torch.stack([s["visual"] for s in samples]),
        "captions": [s["caption"] for s in samples],
    }
