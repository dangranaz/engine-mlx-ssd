# engine-mlx-ssd

**engine-mlx-ssd** is the **SSD expert-streaming (MoE) variant** of engine-mlx — an LLM
inference engine for **Apple Silicon** built on **MLX-C** that adds **SSD-backed expert
offloading** via the `spill` crate. It runs the same **OpenAI-compatible HTTP API** as
engine-mlx, with a `--spill <model>` subcommand to index and stream MoE experts from SSD
**without GPU** — enabling models that exceed unified memory.

> [!IMPORTANT]
> **Correctness first, honesty always.** engine-mlx-ssd inherits engine-mlx's
> token-exact parity with `mlx_lm` at temperature 0, reproducible benchmarks, and
> stable sustained-load behavior. The `spill` crate provides **SSD expert streaming
> for MoE** — mmap + madvise prefetch/release, no external deps. Everything runs
> **locally** on your Mac: private, offline, OpenAI-compatible.

---

## Features

- **Apple Silicon native** — built on Apple MLX via `mlx-c`, links Metal directly.
- **OpenAI-compatible** — `/v1/chat/completions`, `/v1/models`, `/health`, SSE streaming.
- **Token-exact** — matches `mlx_lm` token-for-token at temperature 0 (verified by tests).
- **Static KV cache by default** — pre-allocated buffers + masked SDPA, so decode
  throughput doesn't collapse as context grows.
- **BF16 pipeline**, pipelined `async_eval`, in-graph greedy argmax.
- **Pure-Rust multi-format tokenizer** (BPE / WordPiece / Unigram), HF-parity tested.
- **Stable under sustained load** — each request rebuilds fresh state; no buffer
  accumulation across requests.
- **SSD expert streaming (MoE)** — `spill` crate: mmap + madvise prefetch/release,
  **no external deps**. `nxm-engine-mlx --spill <model>` indexes experts on SSD,
  **no GPU needed** during spill indexing.
- **Portable core** — the non-MLX crates build and test on Linux (stub mode) for CI.
- **Fully self-contained** — 0 git dependencies; `mlx-sys` from crates.io;
  `modelplan` + `tokenizer` inlined as top-level crates.

---

## SSD Expert Streaming (MoE) — the `spill` crate

engine-mlx-ssd includes the `spill` crate — a dedicated SSD expert streaming layer for
Mixture-of-Experts models that exceed unified memory:

- **Mmap-based** — experts memory-mapped from `.safetensors` on disk.
- **Madvise prefetch/release** — hints the kernel to stream pages on demand.
- **No GPU required for indexing** — `nxm-engine-mlx --spill <model>` builds the
  expert index entirely on CPU/SSD.
- **Zero external dependencies** — pure Rust, uses only `mmap` + `madvise`.

```sh
# Index experts to SSD (no GPU; reads model.safetensors, writes spill index)
nxm-engine-mlx --spill ~/models/lmstudio-community/Qwen3-30B-A3B-MLX-4bit

# Then serve normally — experts stream from SSD on demand
nxm-engine-mlx --model ~/models/lmstudio-community/Qwen3-30B-A3B-MLX-4bit
```

---

## Requirements

- **macOS on Apple Silicon** (M-series).
- The MLX C API: `brew install mlx-c`.
- Rust (`cargo`) to build.
- **SSD space** for expert offload (MoE models only).

---

## Quick start

```sh
# 1. install the MLX C API
brew install mlx-c

# 2. build with the `mlx` feature (the build script auto-detects the Homebrew
#    MLX prefixes — no manual paths needed)
cargo build --release --features mlx -p engine-mlx-serve

# 3a. Optional: build SSD spill index for MoE models (no GPU, SSD only)
nxm-engine-mlx --spill ~/models/lmstudio-community/Qwen3-30B-A3B-MLX-4bit

# 3b. run the server on a model directory (must contain config.json)
nxm-engine-mlx --model ~/models/lmstudio-community/Qwen3-1.7B-MLX-4bit
#    → serves http://127.0.0.1:11435
```

Then call it like any OpenAI endpoint:

```sh
curl -s http://127.0.0.1:11435/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3-1.7B-MLX-4bit",
    "messages": [{"role": "user", "content": "Explain quantum computing in one sentence."}],
    "max_tokens": 64,
    "temperature": 0
  }'
```

---

## Getting a model

engine-mlx-ssd does **not** ship model weights — you download MLX-format models
yourself. By default the engine looks for models under **`~/models`**, in the
HuggingFace-style layout:

```
~/models/<org>/<model-name>/
    ├── config.json          # required
    ├── model.safetensors
    └── tokenizer.json
```

For dense models (Qwen3 family): e.g. `~/models/lmstudio-community/Qwen3-1.7B-MLX-4bit/`.
For MoE models (with SSD offload): e.g. `~/models/lmstudio-community/Qwen3-30B-A3B-MLX-4bit/`.

Download with the Hugging Face CLI:

```sh
# install once:  pip install huggingface_hub
huggingface-cli download lmstudio-community/Qwen3-30B-A3B-MLX-4bit \
  --local-dir ~/models/lmstudio-community/Qwen3-30B-A3B-MLX-4bit
```

You can put models anywhere and pass an absolute path, or keep them under
`~/models` and refer to them by `<org>/<name>` (or a bare unique name). Override
the search root with the `ENGINE_MLX_MODELS_DIR` environment variable.

---

## Workspace layout

| Crate        | Role                                                        |
|--------------|-------------------------------------------------------------|
| `mlx-ffi`    | MLX-C bindings (bindgen) + `MlxCtx` real ops / Metal link    |
| `ops`        | Atomic ops (quantized matmul, RoPE, SDPA, RMSNorm, …)        |
| `attention`  | Attention layers (GQA, sliding window, gated)               |
| `kvcache`    | KV cache backends (concat, fp8, rotating)                   |
| `prefill`    | Prefill pipeline (prefix cache + chunked batch prefill)     |
| `serve`      | Qwen3 engine + model loader + OpenAI HTTP server            |
| `modelplan`  | Model introspection → `ModelManifest` (inlined crate)       |
| `tokenizer`  | Pure-Rust multi-format tokenizer, HF-parity (inlined crate) |
| `spill`      | **SSD expert streaming (MoE)** — mmap + madvise, no deps    |

Without the `mlx` feature the workspace builds against stubs — useful for CI on
non-Apple machines and for compiling the non-MLX crates.

---

## Supported models

engine-mlx-ssd targets the **Qwen3** family in MLX format:

| Model              | Type  | Quantization | Status                                  |
|--------------------|-------|--------------|-----------------------------------------|
| Qwen3-0.6B         | dense | 4-bit (MLX)  | ✅ verified, token-exact vs `mlx_lm`    |
| Qwen3-1.7B         | dense | 4-bit (MLX)  | ✅ verified, token-exact vs `mlx_lm`    |
| Qwen3-1.7B         | dense | 8-bit (MLX)  | ✅ verified (native 8-bit via MLX)      |
| Qwen3-30B-A3B      | MoE   | 4-bit (MLX)  | 🔬 SSD spill indexing supported         |

The loader reads `group_size` / `bits` from the model config, so other Qwen3
MLX checkpoints of the same shape should load; only the sizes above are tested.
8-bit compute is **native** via MLX framework (`with_quant(bits=8)`) — no custom
Metal kernel required.

---

## Benchmarks — generation speed

Inherited from engine-mlx (dense models). Real, reproducible numbers live in
[dangranaz/prj-bench](https://github.com/dangranaz/prj-bench). Measured on a
**MacBook Air M1, 16 GB**, greedy (temperature 0):

| Model               | Generation speed | Sustained (20×) | Length ramp 128→1024 |
|---------------------|------------------|-----------------|----------------------|
| Qwen3-1.7B-MLX-4bit | ~34–41 t/s       | ~6% degradation | ~16% drop            |
| Qwen3-0.6B-MLX-4bit | ~55–57 t/s       | —               | —                    |

- **Generation speed** = decode tokens/second.
- **Sustained** = 20 identical requests; throughput must not drift (leak guard).
- **Length ramp** = throughput across 128 / 512 / 1024 output tokens (KV scaling).

`mlx_lm` is still faster in absolute throughput; the gap is kernel efficiency,
not graph overhead. Numbers depend on your chip and thermal state — re-run the
harness to get yours.

> **Note:** MoE models with SSD expert streaming have different latency
> characteristics — expert fetch from SSD adds latency per token. Benchmarks
> for spill-enabled serving will be added as the feature matures.

---

## Honest limitations

- Not a speed record — a correctness-first baseline.
- **No long-context disk offload** for dense models: very large contexts beyond
  RAM are out of scope (dense models must fit in unified memory).
- MoE models with SSD spill trade latency for capacity — expert fetch from SSD
  adds per-token overhead; tune `madvise` hints for your SSD.
- The MLX-C C API doesn't expose the graph fusions of Python `mx.compile`, which
  caps some optimizations.
- Metal-side improvements (q8 shader, load_vector, race-fix) from `engine-metal-ssd`/
  `engine-metal` do **not** transfer: this is an **MLX-C** engine (compute via
  MLX framework, not raw Metal).

---

## Why Rust

Rust was a deliberate choice, for **robustness**: strong typing, no garbage
collector, explicit memory ownership, and errors you handle rather than discover
at runtime — the properties you want in a systems component like an inference
engine. The honest trade-off: for MLX the **Rust ecosystem is still young**
compared to Python — bindings are thinner and examples fewer — so more has to be
built and verified from the primitives. That's a cost, but the robustness (and
the discipline it forces) is worth it here.

---

## How this was built

> The full story — the purpose, the method, the hardware constraint, and the bug
> that proves the point — is in **[STORY.md](./STORY.md)**.

engine-mlx-ssd was recovered from a stub skeleton by syncing the real engine-mlx
implementation while preserving ssD's `spill` crate (SSD MoE streaming). The
engineering that mattered was choosing the right targets, verifying relentlessly,
and reporting results (limitations included) honestly.

**On a tight hardware budget.** All of this was done on a **MacBook Air M1 with
16 GB of unified memory**. That constraint shapes what "works" means, and shows
the engine runs on modest, widely-available hardware.

**Tools.** Built with **OpenCode** and **Pi**, using free AI models — no paid
model subscriptions. The leverage comes from steering agents well and holding
them to a hard bar for "done."

---

## ⭐ Support the project

If engine-mlx-ssd is useful to you — or if you value seeing a working, honestly
benchmarked inference engine with SSD expert streaming built on a modest machine
— please **give the repository a star** and share it. It's the simplest way to
help the project reach other developers. Feedback, issues, and suggestions are
very welcome.

---

## License

MIT — see [LICENSE](./LICENSE).