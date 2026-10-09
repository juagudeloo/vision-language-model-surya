"""train_config.py — Typed configuration for projector training, loaded from a YAML file."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class TrainConfig(BaseModel):
    """Everything a training run needs, so that a launch is fully described by one file.

    Unknown keys are rejected, so a misspelled option fails at launch instead of being silently
    ignored on the cluster.
    """

    model_config = ConfigDict(extra="forbid")

    llm: str = Field(description="Hugging Face id of the frozen language model.")
    llm_precision: Literal["bf16", "8bit", "4bit"] = Field(
        description="How the frozen LLM is loaded. bf16 is unquantized; 4bit is nf4."
    )
    gradient_checkpointing: bool = Field(
        description="Recompute activations in the backward pass: about 45% less peak memory for about 30% less speed."
    )
    manifest: Path = Field(description="JSONL manifest of cached embeddings and captions, relative to the repo root.")
    heldout_manifest: Path | None = Field(
        default=None, description="Optional manifest of samples never trained on; its loss is logged every epoch."
    )
    output_dir: Path = Field(description="Where checkpoints and the loss log are written, relative to the repo root.")
    prompt: str = Field(description="User instruction placed after the image markers.")
    n_visual_tokens: int = Field(gt=0, description="Visual tokens per sample; must match the cached embeddings.")
    batch_size: int = Field(gt=0, description="Samples per forward pass.")
    grad_accum_steps: int = Field(gt=0, description="Forward passes per optimizer step; effective batch is batch_size times this.")
    epochs: int = Field(gt=0, description="Passes over the manifest.")
    learning_rate: float = Field(gt=0, description="AdamW learning rate for the projector.")
    weight_decay: float = Field(ge=0, description="AdamW weight decay.")
    warmup_ratio: float = Field(default=0.03, ge=0, lt=1, description="Share of optimizer steps spent warming the learning rate up; then cosine decay to zero.")
    seed: int = Field(description="Seed for shuffling and projector initialization.")
    log_every: int = Field(gt=0, description="Optimizer steps between loss log lines.")

    @model_validator(mode="after")
    def _check_memory_choice(self) -> "TrainConfig":
        if self.llm_precision == "bf16" and not self.gradient_checkpointing:
            raise ValueError(
                "bf16 without gradient checkpointing only fits at batch 1 on a 21.5 GB card "
                "(19.2 GB peak, measured); enable gradient_checkpointing or choose 4bit/8bit"
            )
        return self

    @property
    def effective_batch(self) -> int:
        """Samples contributing to each optimizer step."""
        return self.batch_size * self.grad_accum_steps


def load_config(path: str | Path) -> TrainConfig:
    """Read and validate a YAML configuration.

    Args:
        path: The YAML file.

    Returns:
        The validated configuration.

    Raises:
        ValueError: If a value is invalid or a key is unknown.
    """
    return TrainConfig.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
