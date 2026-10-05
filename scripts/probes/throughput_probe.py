"""Probe: steady-state throughput and peak VRAM of a projector-training step, on synthetic data.

Usage: python scripts/probes/throughput_probe.py {4bit|8bit|bf16}   (one JSON line per configuration)
Behind the table in CLAUDE.md; rerun it on any other GPU to compare.

Frozen Qwen3-8B in train() mode (so gradient checkpointing really applies), a trainable 22 M
projector, synthetic data shaped like the real thing: 256 visual + 40 prompt + 120 caption tokens.
"""

import gc
import json
import sys
import time

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

QUANT = sys.argv[1]  # 8bit | 4bit | bf16
N_VIS, N_PROMPT, N_CAP = 256, 40, 120
BATCHES = [1, 2, 4, 8, 16]
WARMUP, MEASURE = 1, 3

kwargs = dict(device_map={"": 0}, dtype=torch.bfloat16)
if QUANT == "8bit":
    kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
elif QUANT == "4bit":
    kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
    )

t0 = time.time()
llm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", **kwargs)
for p in llm.parameters():
    p.requires_grad_(False)
llm.train()
print(json.dumps({"event": "loaded", "quant": QUANT, "seconds": round(time.time() - t0), "resident_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
                  "attention_dropout": llm.config.attention_dropout}), flush=True)

hidden = llm.config.hidden_size
projector = nn.Sequential(nn.Linear(1280, hidden), nn.GELU(), nn.Linear(hidden, hidden)).cuda().float()
opt = torch.optim.AdamW(projector.parameters(), lr=1e-4)


def step(bs: int, loss_mode: str) -> float:
    vis = torch.randn(bs, N_VIS, 1280, device="cuda")
    prompt = torch.randint(0, 150000, (bs, N_PROMPT), device="cuda")
    cap = torch.randint(0, 150000, (bs, N_CAP), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        vis_emb = projector(vis).to(torch.bfloat16)
        emb = llm.get_input_embeddings()
        inputs = torch.cat([vis_emb, emb(prompt), emb(cap)], dim=1)
        if loss_mode == "full_logits":
            labels = torch.cat([torch.full((bs, N_VIS + N_PROMPT), -100, device="cuda"), cap], dim=1)
            loss = llm(inputs_embeds=inputs, labels=labels).loss
        else:
            h = llm.model(inputs_embeds=inputs).last_hidden_state
            logits = llm.lm_head(h[:, -(N_CAP + 1):-1])
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), cap.reshape(-1))
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    return loss.item()


for ckpt in (False, True):
    if ckpt:
        llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        llm.gradient_checkpointing_disable()
    for loss_mode in ("full_logits", "supervised_only"):
        for bs in BATCHES:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            oom, row = False, {}
            try:
                for _ in range(WARMUP):
                    step(bs, loss_mode)
                torch.cuda.synchronize()
                t = time.time()
                for _ in range(MEASURE):
                    step(bs, loss_mode)
                torch.cuda.synchronize()
                per_step = (time.time() - t) / MEASURE
                row = dict(quant=QUANT, ckpt=ckpt, loss=loss_mode, batch=bs,
                           peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
                           step_s=round(per_step, 3), samples_per_s=round(bs / per_step, 2),
                           epoch_1000_min=round(1000 / (bs / per_step) / 60, 1))
            except torch.cuda.OutOfMemoryError:
                oom = True
            if oom:
                opt.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
                row = dict(quant=QUANT, ckpt=ckpt, loss=loss_mode, batch=bs, peak_gb="OOM")
            print(json.dumps(row), flush=True)
            if oom:
                break
