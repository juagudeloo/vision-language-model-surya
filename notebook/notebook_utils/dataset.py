"""dataset.py — Load Surya's real config, dataset, and a validation batch in one call.

Wraps the HelioNetCDFDataset + DataLoader setup that sections 5-9 of
docs/surya_forward_pass_study_plan.md build by hand, so later notebooks don't
have to re-copy it. Paths are resolved relative to this file, not the
caller's cwd, so it works the same from any notebook under notebook/.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor
from torch.utils.data import DataLoader

from notebook_utils._paths import DATA_DIR, SURYA_ROOT

from surya.datasets.helio import HelioNetCDFDataset  # noqa: E402
from surya.utils.data import build_scalers, custom_collate_fn  # noqa: E402


class SuryaBatch(BaseModel):
    """Everything a notebook needs after loading Surya's real validation data."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: dict = Field(description="Parsed config.yaml (model + data settings).")
    dataset: HelioNetCDFDataset = Field(description="For indexing more samples than just the loaded one.")
    dataloader: DataLoader = Field(description="Wraps dataset, batch_size=1.")
    sample_index: int = Field(description="Which of the dataset's valid samples batch_data/batch_metadata came from.")
    batch_data: dict[str, Tensor] = Field(
        description='The loaded sample\'s tensors -- "ts" (1, 13, 2, 4096, 4096), "time_delta_input" (1, 2).'
    )
    batch_metadata: dict = Field(description="The loaded sample's metadata (timestamps, etc.), not tensors.")

    @classmethod
    def load(
        cls,
        data_dir: str | Path | None = None,
        index_path: str | Path | None = None,
        phase: str = "valid",
        index: int = 0,
    ) -> "SuryaBatch":
        """Build Surya's real dataset and pull one validation sample, real weights not required.

        Reproduces section 5's setup exactly: real config.yaml/scalers.yaml
        (downloaded in section 3), Surya's own tests/test_surya_index.csv,
        drop_hmi_probability=0 and num_mask_aia_channels=0 (a clean read, not
        the pretraining augmentation), rollout_steps=1 (the dataset needs a
        forecast target to consider a sample valid, even though nothing here
        uses it).

        Args:
            data_dir: Directory holding config.yaml/scalers.yaml. Defaults to
                notebook/data (where section 3 downloads them).
            index_path: CSV listing which timestamps are on disk. Defaults to
                Surya's own tests/test_surya_index.csv.
            phase: Dataset phase passed to HelioNetCDFDataset.
            index: Which of the dataset's valid samples to load as
                batch_data/batch_metadata (0-indexed, dataset order is fixed
                since the dataloader is built with shuffle=False). With the
                real test index this session has been using, valid indices
                are 0-7 (8 samples).

        Returns:
            A SuryaBatch with the config, dataset, dataloader, and the
            requested sample.

        Example:
            >>> from notebook_utils.dataset import SuryaBatch
            >>> batch = SuryaBatch.load(index=3)
            >>> batch.batch_data["ts"].shape
            torch.Size([1, 13, 2, 4096, 4096])
        """
        data_dir = Path(data_dir) if data_dir else DATA_DIR
        index_path = Path(index_path) if index_path else SURYA_ROOT / "tests" / "test_surya_index.csv"

        with open(data_dir / "Surya-1.0" / "config.yaml") as fp:
            config = yaml.safe_load(fp)
        scalers = build_scalers(info=yaml.safe_load(open(data_dir / "Surya-1.0" / "scalers.yaml")))

        dataset = HelioNetCDFDataset(
            index_path=str(index_path),
            time_delta_input_minutes=config["data"]["time_delta_input_minutes"],
            time_delta_target_minutes=config["data"]["time_delta_target_minutes"],
            n_input_timestamps=len(config["data"]["time_delta_input_minutes"]),
            rollout_steps=1,
            channels=config["data"]["sdo_channels"],
            drop_hmi_probability=0.0,
            num_mask_aia_channels=0,
            use_latitude_in_learned_flow=config["data"]["use_latitude_in_learned_flow"],
            scalers=scalers,
            phase=phase,
            pooling=config["data"]["pooling"],
            random_vert_flip=False,
        )

        dataloader = DataLoader(
            dataset, shuffle=False, batch_size=1, num_workers=0,
            pin_memory=False, drop_last=False, collate_fn=custom_collate_fn,
        )
        if not 0 <= index < len(dataset):
            raise IndexError(f"index={index} out of range -- dataset has {len(dataset)} valid sample(s).")
        for i, (batch_data, batch_metadata) in enumerate(dataloader):
            if i == index:
                break

        return cls(
            config=config, dataset=dataset, dataloader=dataloader, sample_index=index,
            batch_data=batch_data, batch_metadata=batch_metadata,
        )
