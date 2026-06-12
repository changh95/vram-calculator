# VRAM Calculator

Answers one question for local-LLM developers: **"Can I run model X on hardware Y?"**

Pick a model, hardware (× quantity), quantization, context length, concurrent
sequences, and serving framework — the page computes required memory (weights +
KV cache + activations + overhead) and verdicts **Fits / Tight / Won't fit**
against the hardware's usable capacity, live on every change. When it won't fit,
it suggests one-click fixes.

A fully client-side React + TypeScript + Tailwind app — all math runs in the
browser, no backend.

## Quick start

```bash
npm install
npm run dev      # local dev server (http://localhost:5173)
npm run build    # static production build → dist/ (host anywhere)
npm test         # Vitest
```

## What makes the math trustworthy

- **Real per-model architecture, not parameter-count heuristics.** KV cache is
  computed from each model's actual config (layers, KV heads, head dim, and the
  attention *kind*): GQA, DeepSeek/Kimi/GLM **MLA** (latent, no ×2), Gemma 4
  **interleaved sliding window**, Qwen3.5/Nemotron **hybrid SSM**, DeepSeek-V4
  **compressed**. Two 32B models can differ 10× in KV — this captures that.
- **Verified data.** Model/hardware/framework values are web-researched against
  HuggingFace configs and vendor spec pages (June 2026) and adversarially
  fact-checked. Every constant traces to a `notes/` doc; estimate-grade terms
  (activations, recurrent state) are flagged in the UI.
- **Honest about estimates.** Weights and KV are accurate; activations and
  runtime overhead are heuristics, disclosed in the "Estimates & assumptions"
  panel. Frameworks that pre-allocate VRAM (vLLM/SGLang/TRT-LLM) are modeled as
  "minimum feasible" with the preallocation noted.

## Layout

```
src/lib/estimateVram.ts   # the formula — pure, isolated, no UI
src/lib/data/             # models.ts, hardware.ts, frameworks.ts, quants.ts
src/lib/{suggestions,urlState,format}.ts
src/components/           # ConfigPanel pieces, Verdict, MemoryGauge, …
notes/                    # vram-formula.md, hardware-constraints.md,
                          #   serving-framework-vram.md (+ raw/ research provenance)
model_list.md · hardware_list.md · serving_framework_list.md
```

## Scope (v1)

Inference only, single-node (plus a simple "× N devices" tensor-parallel model).
Vendor-neutral with first-class Tenstorrent coverage. Out of scope: training/
fine-tuning memory, accounts, explicit pipeline/expert parallelism, CPU/disk
offload math.

See `CLAUDE.md` for the full architecture and contribution rules — data tables
and the `notes/`/list docs must never drift apart.
