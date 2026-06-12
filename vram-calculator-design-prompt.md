# Design Prompt — VRAM Calculator

> Paste this into an AI builder (v0, Lovable, Bolt, Claude). It describes **what to build and how it should look and behave**. Technical unknowns are listed at the end as open decisions — do not invent values for them; stub them and surface the assumption in the UI.

---

## 1. One-line concept

A single-page web app that answers one question for local-LLM developers: **"Can I run model X on hardware Y?"** The user picks a model and a configuration, and the page calculates required VRAM and verdicts whether the chosen hardware fits.

## 2. Target user

Local-LLM developers and hobbyists — technically literate, impatient, comparing options. They want a fast answer and a clear breakdown they can trust, not a wall of prose. They will tweak inputs repeatedly to find a config that fits, so the app must recalculate instantly and make trade-offs legible.

## 3. Core user flow

1. User lands on the page; sensible defaults are pre-selected so a verdict is visible immediately (no empty state).
2. User adjusts inputs (model, hardware, quantization, context length, batch size, concurrent users, serving framework).
3. The verdict and memory breakdown **update live** on every change — no "Calculate" button required (but include one as a fallback affordance for mobile).
4. User can compare the same model across multiple hardware targets, or sweep one knob to see where it stops fitting.

## 4. Inputs (controls)

Group controls into a left-hand or top **configuration panel**. Each control:

- **Model** — searchable dropdown / combobox. Families: Qwen3, Gemma 4, GLM-5, DeepSeek, Kimi, Nemotron. Show parameter count next to each variant (e.g. "Qwen3-32B"). Group by family.
- **Hardware** — searchable dropdown grouped by vendor, with each option showing its memory capacity:
  - **NVIDIA:** RTX 2080 Ti, RTX 3080/3090, RTX 4080/4090, RTX 5060/5070/5080/5090, A6000, RTX Pro 6000, L40S, A100, H100, B200, B300, DGX Spark
  - **Apple:** M1/M2/M3/M4/M5 × Base/Pro/Max/Ultra (note: unified memory — see open questions)
  - **AMD:** MI300X, MI350X (+ room to add more)
  - **Tenstorrent:** N150, N300, T3K, WH Galaxy, P150, QB2, BH Galaxy
  - Allow selecting **multiple** hardware targets for side-by-side comparison.
- **Quantization** — segmented control or dropdown: FP16/BF16, FP8, INT8, INT4 (GGUF Q4), and other common GGUF/AWQ/GPTQ levels. Show bytes-per-weight implied by each.
- **Context length** — slider + numeric input (e.g. 1K → 1M tokens), with common presets (4K, 8K, 32K, 128K).
- **Batch size** — numeric stepper.
- **Concurrent users** — numeric stepper (drives how many KV-cache copies / total context).
- **Serving framework** — dropdown: vLLM, SGLang, llama.cpp, Ollama, TensorRT-LLM, MLX (Apple), etc. Framework choice affects KV-cache strategy and overhead (see open questions).

Every control needs an inline tooltip explaining how it affects memory.

## 5. Output (the answer)

The right-hand / main **results area** is the hero of the page. It must contain:

- **Verdict badge** — large, unmistakable. Three states:
  - ✅ **Fits** (green) — required VRAM comfortably under capacity.
  - ⚠️ **Tight** (amber) — fits but within a small safety margin (e.g. >90% utilization).
  - ❌ **Won't fit** (red) — required VRAM exceeds capacity. Show the shortfall ("needs 38 GB, you have 24 GB — over by 14 GB").
- **Required vs. available bar** — a horizontal stacked bar showing total required VRAM against the hardware's capacity line, so over/under is visible at a glance.
- **Memory breakdown** — stacked breakdown of where the memory goes, each as its own segment with a number:
  - Model weights
  - KV cache (scales with context × batch × concurrent users)
  - Activations / overhead
  - Framework / runtime overhead
  Render as a stacked bar **and** an itemized table.
- **Suggestions when it won't fit** — actionable nudges: "Drop to INT4 to save ~X GB", "Reduce context to 32K", "This fits on an H100 instead." (Logic depends on open questions; structure the UI to accept these.)
- **Comparison view** — when multiple hardware targets are selected, show a compact table/grid: rows = hardware, columns = required VRAM, capacity, verdict.

## 6. Layout

- Desktop: two-pane — **config panel (left, ~⅓)** and **results (right, ~⅔)**. Results stay in view while scrolling the config.
- Mobile: stacked — config collapses into an accordion or sheet; results pinned at top with the verdict badge.
- Keep the whole default experience above the fold on desktop. No multi-step wizard.

## 7. Visual style

- Clean, technical, developer-tool aesthetic — think a well-designed dashboard, not a marketing page. Reference feel: Vercel / Linear / a good benchmarking tool.
- **Dark mode default**, with light mode toggle.
- Monospace for numbers and units (GB, tokens); sans-serif for labels.
- Color is meaningful, not decorative: green/amber/red reserved strictly for the verdict and the utilization bar. Neutral grays elsewhere.
- Dense but breathable. Numbers are first-class — large, aligned, easy to scan.
- Subtle transitions when values recalculate (animate the bars) so changes feel responsive, not jarring.

## 8. Behavior & states

- **Live recalculation** on every input change, debounced.
- **Shareable state** — encode the full config in the URL so a configuration can be linked.
- **Empty/default state** — never blank; load with a popular default (e.g. Qwen3-8B on RTX 4090, INT4, 8K context).
- **Assumption disclosure** — wherever the math relies on an undecided formula or estimate, show an "ℹ️ estimate" marker and a note. Trust depends on honesty about approximations.
- Accessibility: verdict must not rely on color alone (icon + text label), all controls keyboard-navigable, contrast meets WCAG AA.

## 9. Out of scope (v1)

Login/accounts, saving configs server-side, fine-tuning/training memory (inference only), and multi-GPU sharding math (note it as "coming later" rather than faking it).

---

## 10. Open technical questions — DO NOT GUESS

These were not yet decided. The builder should **stub these as clearly-labeled, swappable functions/constants** and surface "estimate" markers in the UI rather than inventing authoritative numbers.

1. **Tech stack** — not chosen. Reasonable default for an AI-builder target: React + TypeScript + Tailwind, fully client-side (all math runs in the browser, no backend needed). Flag if you assume otherwise.
2. **VRAM calculation formula** — the exact formula for total VRAM = f(weights, KV cache, activations, overhead) is undecided. Implement it as a single isolated, documented module (`estimateVram(config)`) with each term as a named, replaceable function so the formula can be corrected without touching the UI.
3. **Unified-memory limits (Apple, DGX Spark, etc.)** — these share RAM between system and GPU, and the usable fraction is capped (e.g. macOS reportedly allows ~75% of total RAM for GPU/LLM use). The exact per-platform cap is unconfirmed — parameterize it per device and label the cap in the UI.
4. **VRAM-saving techniques & KV-cache optimizations** — how much each serving framework saves (paged attention, KV-cache quantization, FP8 KV cache, prefix caching, etc.) and by how much is undecided. Model these as per-framework modifier factors in a config table, clearly marked as estimates.
5. **Quantization byte-per-weight values** — confirm the exact effective bytes/parameter for each quant scheme (including overhead for scales/zero-points in GGUF/AWQ/GPTQ) rather than assuming clean 4/8/16-bit.

Wherever one of these is used, the UI should make clear the number is an estimate, not a guarantee.
