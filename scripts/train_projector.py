#!/usr/bin/env python3
"""train_projector.py — Train the projector with Surya and the language model both frozen.

Each sample is a region's cached (256, 1280) Surya tokens plus a caption. The tokens go through the
projector, replace the image markers in the chat text, and the loss is the next-token loss on the
caption and its end marker only. Only the projector's ~22 M parameters get gradients.

The three sequence functions (`build_example`, `collate`, `inject_visual`) live in `sequence.py`; see
section 10 of docs/surya_forward_pass_study_plan.md for their contract.

Usage:
  python scripts/train_projector.py --config configs/prototype.yaml
"""

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from cached_dataset import CachedEmbeddingDataset, collate_samples  # noqa: E402
from projector import Projector  # noqa: E402
from sequence import build_example, collate, inject_visual  # noqa: E402
from train_config import TrainConfig, load_config  # noqa: E402


def load_llm(config: TrainConfig) -> AutoModelForCausalLM:
    """Load the frozen language model ready for training the projector through it.

    The model is left in train() mode on purpose: Hugging Face only applies gradient checkpointing in
    train mode. Qwen3 has no dropout (attention_dropout is 0.0), so this changes no numbers.

    Args:
        config: The run configuration.

    Returns:
        The model on the GPU with every parameter frozen.
    """
    kwargs: dict = {"device_map": {"": 0}, "dtype": torch.bfloat16}
    if config.llm_precision == "8bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif config.llm_precision == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )
    llm = AutoModelForCausalLM.from_pretrained(config.llm, **kwargs)
    for parameter in llm.parameters():
        parameter.requires_grad_(False)
    llm.config.use_cache = False
    llm.train()
    if config.gradient_checkpointing:
        llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return llm


def caption_loss(llm, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Next-token loss on the supervised positions only, never building logits for the rest.

    Equivalent to passing `labels` to the model, but the output head is applied only where a label
    exists, which saves memory because the vocabulary has about 152,000 entries.

    Args:
        llm: The causal language model.
        inputs_embeds: Shape (B, S, D), with the image tokens already injected.
        attention_mask: Shape (B, S), 1 for real tokens.
        labels: Shape (B, S), -100 wherever no loss is wanted.

    Returns:
        The mean cross-entropy over supervised tokens, a scalar float32 tensor.
    """
    hidden = llm.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).last_hidden_state
    shifted = labels[:, 1:]
    keep = shifted != -100
    logits = llm.lm_head(hidden[:, :-1][keep])
    return F.cross_entropy(logits.float(), shifted[keep])


def warmup_cosine(config: TrainConfig, batches_per_epoch: int):
    """Learning-rate multiplier: linear warmup, then cosine decay to zero, as in LLaVA's training.

    Args:
        config: The run configuration.
        batches_per_epoch: Batches in one pass over the training manifest.

    Returns:
        A function from optimizer step number to a multiplier in [0, 1], for LambdaLR.
    """
    total = math.ceil(batches_per_epoch / config.grad_accum_steps) * config.epochs
    warmup = max(1, round(config.warmup_ratio * total))

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return factor


def check_gradients(llm, projector: Projector) -> None:
    """Fail loudly if the gradient does not reach the projector or leaks into the frozen model.

    Args:
        llm: The frozen language model.
        projector: The projector.

    Raises:
        RuntimeError: If any language-model parameter has a gradient, or the projector has none.
    """
    leaked = [n for n, p in llm.named_parameters() if p.grad is not None]
    if leaked:
        raise RuntimeError(f"gradient reached the frozen LLM: {leaked[:3]}")
    if not any(p.grad is not None and p.grad.abs().sum() > 0 for p in projector.parameters()):
        raise RuntimeError("no gradient reached the projector")
    print("gradient check: projector has gradient, frozen LLM has none", flush=True)


def make_loader(manifest: Path, config: TrainConfig, tokenizer, shuffle: bool, seed: int) -> DataLoader:
    """Build a loader that yields ready-to-run batches.

    Args:
        manifest: JSONL manifest, relative to the repo root.
        config: The run configuration.
        tokenizer: The language model's tokenizer.
        shuffle: Whether to shuffle each epoch.
        seed: Seed for the shuffling order.

    Returns:
        A DataLoader whose batches hold `visual`, `input_ids`, `labels` and `attention_mask` tensors.
    """
    dataset = CachedEmbeddingDataset(REPO / manifest)

    def to_batch(samples: list[dict]) -> dict:
        raw = collate_samples(samples)
        if raw["visual"].shape[1] != config.n_visual_tokens:
            raise ValueError(f"cached tokens {raw['visual'].shape[1]} != n_visual_tokens {config.n_visual_tokens}")
        examples = [build_example(tokenizer, c, config.n_visual_tokens, config.prompt) for c in raw["captions"]]
        input_ids, labels, attention_mask = collate(tokenizer, examples)
        return {"ids": raw["ids"], "visual": raw["visual"], "input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle, collate_fn=to_batch, generator=generator)


def run_batch(llm, projector: Projector, batch: dict, device: torch.device) -> torch.Tensor:
    """Forward one batch and return its loss.

    Args:
        llm: The frozen language model.
        projector: The trainable projector.
        batch: A batch from make_loader.
        device: Where the tensors go.

    Returns:
        The caption loss, a scalar that carries gradient to the projector.
    """
    with torch.autocast(device.type, dtype=torch.bfloat16):
        visual = projector(batch["visual"].to(device).float())
        embeds = inject_visual(llm, batch["input_ids"].to(device), visual)
        return caption_loss(llm, embeds, batch["attention_mask"].to(device), batch["labels"].to(device))


@torch.no_grad()
def evaluate(llm, projector: Projector, loader: DataLoader, device: torch.device) -> float:
    """Mean caption loss over a loader, with no gradient.

    Args:
        llm: The frozen language model.
        projector: The projector to evaluate.
        loader: Batches to score.
        device: Where the tensors go.

    Returns:
        The loss averaged over batches.
    """
    losses = [run_batch(llm, projector, batch, device).item() for batch in loader]
    return sum(losses) / len(losses)


def main() -> None:
    """Parse arguments, train, log every optimizer step, and save the projector."""
    parser = argparse.ArgumentParser(description="Train the projector with Surya and the LLM frozen.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-optimizer-steps", type=int, default=None, help="Stop early (smoke tests).")
    args = parser.parse_args()
    config = load_config(args.config)

    torch.manual_seed(config.seed)
    device = torch.device("cuda")
    out_dir = REPO / config.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out_dir / "config.yaml")

    tokenizer = AutoTokenizer.from_pretrained(config.llm)
    llm = load_llm(config)
    projector = Projector(1280, llm.config.hidden_size).to(device)
    optimizer = torch.optim.AdamW(projector.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    train_loader = make_loader(config.manifest, config, tokenizer, shuffle=True, seed=config.seed)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine(config, len(train_loader)))
    heldout_loader = make_loader(config.heldout_manifest, config, tokenizer, False, config.seed) if config.heldout_manifest else None
    print(f"train samples {len(train_loader.dataset)}, held out {len(heldout_loader.dataset) if heldout_loader else 0}, "
          f"effective batch {config.effective_batch}, trainable parameters {sum(p.numel() for p in projector.parameters()) / 1e6:.2f} M",
          flush=True)

    log = open(out_dir / "train_log.jsonl", "w")
    step, started, best = 0, time.time(), float("inf")
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, config.epochs + 1):
        window: list[float] = []
        for i, batch in enumerate(train_loader, start=1):
            loss = run_batch(llm, projector, batch, device)
            (loss / config.grad_accum_steps).backward()
            if step == 0 and i == 1:
                check_gradients(llm, projector)
            window.append(loss.item())
            if i % config.grad_accum_steps == 0 or i == len(train_loader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % config.log_every == 0:
                    record = {"type": "train", "epoch": epoch, "step": step, "loss": sum(window) / len(window), "lr": scheduler.get_last_lr()[0],
                              "seconds": round(time.time() - started, 1), "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                    print(json.dumps(record), flush=True)
                window = []
                if args.max_optimizer_steps and step >= args.max_optimizer_steps:
                    break
        if heldout_loader:
            held = evaluate(llm, projector, heldout_loader, device)
            log.write(json.dumps({"type": "eval", "epoch": epoch, "step": step, "heldout_loss": held}) + "\n")
            log.flush()
            print(json.dumps({"type": "eval", "epoch": epoch, "heldout_loss": round(held, 4)}), flush=True)
            if held < best:
                best = held
                torch.save(projector.state_dict(), out_dir / "projector_best.pt")
        torch.save(projector.state_dict(), out_dir / "projector_last.pt")
        if args.max_optimizer_steps and step >= args.max_optimizer_steps:
            break
    log.close()
    print(f"done: {step} optimizer steps in {time.time() - started:.0f}s; saved to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
