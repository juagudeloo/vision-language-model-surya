#!/usr/bin/env python3
"""generate_captions.py — Caption cached samples with a trained projector and print them next to the reference.

Builds the same prompt the model was trained with (image markers, instruction, empty think block), puts
the projector's output where the markers are, and decodes greedily.

Usage:
  python scripts/generate_captions.py --config configs/prototype.yaml --checkpoint runs/prototype/projector_best.pt \\
      --manifest data/cache/manifest_heldout.jsonl --out runs/prototype/captions_heldout.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from cached_dataset import CachedEmbeddingDataset  # noqa: E402
from projector import Projector  # noqa: E402
from sequence import build_example, inject_visual  # noqa: E402
from train_config import load_config  # noqa: E402
from train_projector import load_llm  # noqa: E402


def prompt_ids(tokenizer, config) -> list[int]:
    """Token ids of everything before the caption, exactly as in training.

    Uses build_example with a placeholder caption and keeps the unsupervised prefix, so this cannot
    drift from what training saw.

    Args:
        tokenizer: The language model's tokenizer.
        config: The run configuration.

    Returns:
        The ids up to and including the empty think block that opens the assistant turn.
    """
    ids, labels = build_example(tokenizer, "placeholder", config.n_visual_tokens, config.prompt)
    first_supervised = next(i for i, label in enumerate(labels) if label != -100)
    return ids[:first_supervised]


@torch.no_grad()
def main() -> None:
    """Generate a caption for each sample of a manifest and write reference and generated text."""
    parser = argparse.ArgumentParser(description="Caption cached samples with a trained projector.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)

    tokenizer = AutoTokenizer.from_pretrained(config.llm)
    llm = load_llm(config)
    llm.eval()
    llm.config.use_cache = True
    if config.gradient_checkpointing:
        llm.gradient_checkpointing_disable()
    device = torch.device("cuda")
    projector = Projector(1280, llm.config.hidden_size).to(device)
    projector.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    projector.eval()

    dataset = CachedEmbeddingDataset(REPO / args.manifest)
    ids = torch.tensor([prompt_ids(tokenizer, config)], device=device)
    results = []
    for index in range(min(len(dataset), args.limit or len(dataset))):
        sample = dataset[index]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            visual = projector(sample["visual"].unsqueeze(0).to(device).float())
            embeds = inject_visual(llm, ids, visual)
            generated = llm.generate(
                inputs_embeds=embeds, attention_mask=torch.ones(embeds.shape[:2], device=device, dtype=torch.long),
                max_new_tokens=args.max_new_tokens, do_sample=False,
            )
        text = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        results.append({"id": sample["id"], "reference": sample["caption"], "generated": text})
        print(f"[{index + 1}] {sample['id']}\n   reference: {sample['caption']}\n   generated: {text}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results), encoding="utf-8")
    print(f"{len(results)} captions -> {args.out}")


if __name__ == "__main__":
    main()
