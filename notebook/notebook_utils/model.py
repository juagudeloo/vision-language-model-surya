"""model.py — Build Surya, load its real weights, and run the encoder forward pass.

Wraps the HelioSpectFormer construction (section 3) and weight loading +
embedding forward pass (section 5) that
docs/surya_forward_pass_study_plan.md builds by hand, so later notebooks
don't have to re-copy it.
"""

from pathlib import Path

import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor

from notebook_utils._paths import DATA_DIR

from surya.models.helio_spectformer import HelioSpectFormer  # noqa: E402


class SuryaModel(BaseModel):
    """Surya's real architecture with its pretrained weights loaded.

    embed_batch() is a method rather than a free function because it always
    needs this exact model and this exact device together -- bundling them
    means a call site can't accidentally mix a model with the wrong device,
    and doesn't have to pass either explicitly.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: HelioSpectFormer = Field(
        description="HelioSpectFormer with real pretrained weights, moved to device, in eval mode."
    )
    device: torch.device = Field(description="Device the model was moved to (cuda if available, else cpu).")
    config: dict = Field(description="Parsed config.yaml used to build the architecture.")

    @classmethod
    def load(
        cls,
        config: dict | None = None,
        data_dir: str | Path | None = None,
        device: str | torch.device | None = None,
    ) -> "SuryaModel":
        """Build Surya's architecture and load its real pretrained weights, ready for inference.

        Reproduces sections 3 and 5's setup exactly: HelioSpectFormer built
        from config.yaml's real values, then surya.366m.v1.pt loaded with
        strict=True, moved to device, and set to eval() -- weights are not
        frozen here (no requires_grad=False), since that's a training-time
        decision made in section 9, not part of just loading the model.

        Args:
            config: Parsed config.yaml. If None, loaded fresh from data_dir
                (the same file load_surya_batch reads, so pass its .config
                to avoid reading it twice).
            data_dir: Directory holding config.yaml/surya.366m.v1.pt.
                Defaults to notebook/data (where section 3 downloads them).
                Ignored if config is given directly, except for locating
                the checkpoint.
            device: Defaults to "cuda" if available, else "cpu".

        Returns:
            A SuryaModel with the ready-to-use model, its device, and config.

        Example:
            >>> from notebook_utils.model import SuryaModel
            >>> surya = SuryaModel.load()
            >>> sum(p.numel() for p in surya.model.parameters()) / 1e6
            366.2
        """
        data_dir = Path(data_dir) if data_dir else DATA_DIR
        if config is None:
            with open(data_dir / "Surya-1.0" / "config.yaml") as fp:
                config = yaml.safe_load(fp)
        device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = HelioSpectFormer(
            img_size=config["model"]["img_size"],
            patch_size=config["model"]["patch_size"],
            in_chans=len(config["data"]["sdo_channels"]),
            embed_dim=config["model"]["embed_dim"],
            time_embedding={
                "type": "linear",
                "time_dim": len(config["data"]["time_delta_input_minutes"]),
            },
            depth=config["model"]["depth"],
            n_spectral_blocks=config["model"]["n_spectral_blocks"],
            num_heads=config["model"]["num_heads"],
            mlp_ratio=config["model"]["mlp_ratio"],
            drop_rate=config["model"]["drop_rate"],
            window_size=config["model"]["window_size"],
            dp_rank=config["model"]["dp_rank"],
            learned_flow=config["model"]["learned_flow"],
            rpe=config["model"]["rpe"],
            ensemble=config["model"]["ensemble"],
            finetune=config["model"]["finetune"],
        )
        model.to(device)
        weights = torch.load(data_dir / "Surya-1.0" / "surya.366m.v1.pt", map_location=device)
        model.load_state_dict(weights, strict=True)
        model.eval()

        return cls(model=model, device=device, config=config)

    def embed_batch(self, batch_data: dict[str, Tensor]) -> Tensor:
        """Run Surya's embedding + backbone forward pass, no_grad, skipping unembed.

        This is *not* the same as calling self.model(batch) -- with this
        checkpoint's finetune=False, that returns unembed's reconstructed
        image, not the token embedding. Calling embedding/backbone directly
        is the only way to get the (1, 65536, 1280) tokens themselves. See
        section 6 of docs/surya_forward_pass_study_plan.md for the full
        explanation.

        Args:
            batch_data: A batch dict with "ts" and "time_delta_input" keys,
                as returned by load_surya_batch().batch_data.

        Returns:
            The encoder's token embedding, shape (1, 65536, 1280).

        Example:
            >>> from notebook_utils.dataset import load_surya_batch
            >>> from notebook_utils.model import SuryaModel
            >>> batch = load_surya_batch()
            >>> surya = SuryaModel.load(config=batch.config)
            >>> tokens = surya.embed_batch(batch.batch_data)
            >>> tokens.shape
            torch.Size([1, 65536, 1280])
        """
        with torch.no_grad():
            tokens = self.model.embedding(
                batch_data["ts"].to(self.device), batch_data["time_delta_input"].to(self.device)
            )
            tokens = self.model.backbone(tokens)
        return tokens
