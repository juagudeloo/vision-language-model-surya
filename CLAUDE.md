# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A vision-language model for **captioning solar observations**: NASA and IBM's Surya as a frozen
vision encoder, a trainable MLP projector, and an LLM decoder. PhD research (year 2, Universidad
Nacional de Colombia).

The repo is **empty except for this file** — no code has been written yet. Everything below is the
architectural premise, not a description of existing code. Do not infer that a module exists
because it is named here.

Training data comes from a sibling project, `SDO_paper_to_observations`, which extracts solar
figures from papers and re-queries SDO for the matching observation. That pipeline is **not a
prerequisite** (co-advisor's explicit directive): the prototype starts from ~20 images with
LLM-generated captions that already exist from Julio Flórez's work.

The parent directory `Doctorado/` (untracked, local only) carries cross-cutting context and its own
`CLAUDE.md`, which loads automatically alongside this one. The authoritative milestone document is
`../context/hito_arquitectura_VLM.md` — **re-read it before substantive work**; it outranks this
file on detail.

## Architecture

```
solar observation → [Surya encoder, frozen] → [MLP projector, trainable] → [LLM decoder] → caption
```

The projector is a **non-linear** MLP with GELU:

```
v' = W₂ · GELU(W₁ v + b₁) + b₂
```

The non-linearity is a deliberate choice, not a default. Surya was pretrained on **forecasting**
(one-hour-ahead prediction with autoregressive rollout), not on contrastive image-text alignment.
Its embedding space was never shaped to be linearly comparable with text, so a single linear
projection has less reason to suffice here than in the SigLIP2-style recipes the literature
describes. Only the projector trains; Surya's weights stay frozen (`requires_grad=False`).

## Encoder shapes and the projection decision (resolved 27 Aug 2026)

Verified against the checkpoint's `config.yaml` and the code in `NASA-IMPACT/Surya`
(`helio_spectformer.py`, `embedding.py`) — not the paper alone.

| | |
|---|---|
| `img_size` / `patch_size` | 4096 / 16 |
| `embed_dim` | **1280** |
| `depth` / `num_heads` | 10 (2 spectral + 8 long-short attention) / 16 |
| `n_input_timestamps` | **2**, `time_delta_input_minutes: [-60, 0]` |
| channels | 13 (8 AIA + 5 HMI) |

**Input:** `ts` of shape `(B, C, T, H, W)` = `(B, 13, 2, 4096, 4096)`.
**Encoder output:** `(B, L, D)` = `(B, 65536, 1280)` — **no temporal axis.**

The patch embedding is `nn.Conv2d(in_chans * time_dim, embed_dim, kernel=16, stride=16)` applied
after `x.flatten(1, 2)`, which collapses channel and time into one axis: `(B, 13, 2, H, W)` →
`(B, 26, H, W)`. **Time enters as channels, not as a sequence axis.** So there is no temporal
dimension to flatten or pool at the projector — the encoder already folded it in.

The real problem is spatial: **65,536 visual tokens** where LLaVA-1.5 uses 576. That exceeds the
native context window of both candidate LLMs on its own.

**Decision — token-cropping, then uniform spatial pooling:**

1. Run Surya on the **full disk** (in-distribution; it never saw crops in pretraining) and select
   only the tokens covering the region of interest. Each token is one 16×16 pixel patch, so a pixel
   bbox maps to token indices by integer division by 16. A 512×512 px active region is 32×32 = 1024
   tokens.
2. Uniformly pool those already-cropped tokens down to ~256.

Do **not** crop the image before the encoder: position embeddings are tied to the 256×256 grid, the
spectral gating operates via FFT (not obviously size-invariant), and a crop destroys the scale and
disk-position cues the model learned. Cropping tokens rather than pixels also avoids pooling the
active region together with empty disk, which was the one serious objection to uniform pooling.

Better reducers (perceiver resampler, learned conv over the token grid) are deferred to Sprint 3,
once there are metrics to show they help.

### Consequences

- **Encoder input must come from the SDO archive, not from paper PNGs.** Surya requires 13 channels
  × 2 timestamps × full disk. A published figure is one moment, one or two channels, often a crop —
  it cannot be fed to Surya. Valid input comes from re-querying VSO, which is what stage 4 of
  `SDO_paper_to_observations` does.
- **Token-cropping multiplies the dataset**: one observation with three active regions yields three
  training samples. That matters at ~20 observations.
- The pipeline's `metadata` stage already emits `panels[]` with `active_region` and
  `heliographic_location`. The chain is heliographic → pixel (sunpy `Map.world_to_pixel`, using the
  FITS WCS) → token index (÷16).
- `time_delta_input` is accepted by `forward` but **not used**. The model cannot know the temporal
  spacing it is given and assumes the pretraining one. Keep any multi-frame scheme near 60 minutes.
- **Flare dynamics are not resolvable.** AIA images every 12 s; Surya was trained on a 12-minute
  harmonized cadence and takes two frames 60 minutes apart. It gives before/after, not the impulsive
  phase. Keep this in mind when choosing which phenomena enter the pilot dataset.

### If more than 2 timestamps are ever needed

`time_dim` appears in exactly one place — that Conv2d's `in_channels` — and changing it does **not**
change the output shape. Preferred path: run Surya multiple times over frame pairs and concatenate,
which keeps the encoder frozen and cacheable, and is the only situation where a temporal axis
actually reaches the projector. The alternative is inflating the patch embedding weights from
`(1280, 26, 16, 16)` to `(1280, 52, 16, 16)` (I3D-style: replicate and scale by N/M) and fine-tuning
that layer, which costs the fully-frozen-encoder property for ~8.5M of 366M parameters.

## Surya reference (verified against the paper and model card)

- 366M-parameter **spatiotemporal transformer** with spectral gating and long-short range attention.
  It is *not* a plain ViT; that distinction is the reason the decision above exists.
- Pretrained on ~9 years (≈218 TB) of SDO data: 8 AIA channels + 5 HMI products, at native
  4096×4096 resolution with a 12-minute cadence.
- Pretraining objective: one-hour-ahead forecasting plus autoregressive rollout tuning.
- Weights: `nasa-ibm-ai4science/Surya-1.0` on Hugging Face. Code: `NASA-IMPACT/Surya` on GitHub.
  Paper: arXiv:2508.14112.

Andrés Muñoz-Jaramillo (SwRI), the co-advisor on this project, is a project scientist for Surya.

## Unknown / undecided

Do not guess at these — a wrong assumption propagates into every shape downstream.

- **Migrating the rest of the stack to Qwen3-8B.** The decision (25 Sep 2026) is one model family
  for the whole thesis: the VLM decoder, the Level-2 judge, the pipeline's `metadata` stage and the
  agent. Not done yet: `SDO_paper_to_observations` still runs Qwen2.5-14B-Instruct at 8-bit, and
  its metadata-extraction quality must be re-validated after the swap.
- **What Julio Flórez's ~20 images actually are.** If they are paper figures rather than archive
  observations, they cannot be fed to Surya as-is (see Consequences above). Verify before Sprint 1.

The projector's `in_dim` is **1280**, resolved above. Its `out_dim` is **4096**: the LLM decoder is
**Qwen3-8B** (decided 25 Sep 2026), whose `hidden_size` is 4096 (verified in its Hugging Face
`config.json`; Qwen2.5-14B-Instruct would have been 5120). Rationale and sources are in
`../context/hito_arquitectura_VLM.md`. Not yet measured: VRAM of a real projector-training step.

## Hardware

Work runs on the `fisica` SLURM cluster, node `maxwell` (RTX A4500, 20 GB, shared). Batch launchers
should follow the pattern in `../../MUISCA/tools/*.sh` (`--cluster=fisica`, `-w maxwell`,
`--partition=gpu.cecc`, `--gres=gpu:1`) once full training starts; interactive is fine while
prototyping.

Memory budget, so it is not re-derived: the input tensor `(1, 13, 2, 4096, 4096)` is **1.74 GB in
fp32** (872 MB fp16) before any activations. Peak inference with `no_grad()` at batch 1 is roughly
**4–8 GB** — fits in 20 GB, batch 2–4 gets tight.

That cost is paid **once**. With the encoder frozen, precompute the embeddings, cache them, and the
projector's training loop never loads Surya at all — encoder and LLM never coexist in VRAM. Cached
after reduction, each observation is ~655 KB (256 tokens × 1280, fp16). Raw archive data is the real
storage cost: ~1.74 GB per observation.

If a design starts being shaped by the 20 GB ceiling, say so rather than silently scaling it down —
a larger node (`nodo-01`, 2× AMD MI210) exists, and more can be requested from the cluster admin.

## Constraints

- **Open-source models only** in the runtime stack. No paid or proprietary model APIs — budget plus
  the project's open-sourceable positioning. This does not bear on development tooling.
- Evaluation is designed in three levels: automatic lexical metrics (BLEU/METEOR/ROUGE-L/CIDEr),
  then an LLM-as-judge scoring physical coherence (correct phenomenon, correct AIA/HMI channel),
  then human review of a sample. Reuse this scheme rather than re-solving evaluation.

## Conventions

Sprints 1–2 are **deliberately exploratory**: notebooks, no formal experiment tracking, no CI/CD, no
package structure. This is intentional and comes from the milestone document — do not restructure
this repo into a package or add tracking before the prototype is validated. Experiment tracking
(MLflow) enters in Sprint 3, when there are checkpoints worth comparing.

## Commands

None yet — there is no code, no environment file, and no test suite. Add them here as they appear,
along with how to launch on the PhD cluster.
