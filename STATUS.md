# STATUS — engine-mlx-ssd

## State: RECOVERED → independent + functional
`engine-mlx-ssd` is an independent (self-contained) fork of `engine-mlx`, focused on
Metal/Apple-Silicon inference with **SSD expert streaming (MoE)** via the `spill` crate.

- Independence: **0 git deps** (`Cargo.lock`: no `git+` sources); MLX-C via `mlx-sys`
  (crates.io) + inlined `crates/modelplan` + `crates/tokenizer`. Everything lives in-repo
  (the old `external/` vendored snapshot is superseded by the inlined crates).
- Build: `cargo check --workspace --all-targets` ✅ (macOS arm64, Xcode/clang 5.0).

## History of this repo
1. `6b6b545` — *original stub*: vendored `external/` deps, stub `ops`/`ffi`/`linear`
   (the old `ops/lib.rs` was literally "Stub implementation — Full implementation
   requires MLX-C headers").
2. `feat(mlxs): recover real MLX ops/serve impl from engine-mlx; preserve SSD spill` —
   replaces the stub crates with the real `engine-mlx` implementation + keeps the `spill`
   crate (ssD's real contribution) + adds the `--spill` CLI subcommand. Non-destructive
   (this commit is a child of `6b6b545`).

## What works (recovered from live engine-mlx)
- 8 crates: `ops`, `mlx-ffi`, `attention`, `prefill`, `kvcache`, `serve`, `modelplan`,
  `tokenizer` + ssD's `spill`.
- `mlx-ffi`: real bindgen bindings over `mlx-c`/`mlx`; `MlxCtx` with real ops
  (`mlx_quantized_matmul`, `mlx_fast_rope`, `mlx_fast_sdpa`, `mlx_dequantize`,
  `mlx_rms_norm`, …) + cfg-gated stubs.
- `ops`: real MLX compute — `quant` (`QuantWeights` w/ `qmatmul` → Q4/mxfp4/**Q8** via
  `with_quant(bits=8)`), `embed`, `fp8`, `gdn`, `mlp`, `rope`, `lm_head`, `moe`, …
  (8-bit compute is native via the MLX framework — no custom Metal kernel needed).
- `serve`: `Qwen3Engine::load` → `embed` → `prefill` → `decode_step`; token-exact vs
  `mlx_lm` at temperature 0; OpenAI-compatible HTTP + SSE; stable under sustained load.
- `spill` (ssD, preserved): SSD MoE expert streaming — mmap + madvise prefetch/release,
  no external deps. `nxm-engine-mlx --spill <model>` indexes & streams experts from SSD.

## Notes / decisions
- ssD's old stub-only files (`ops/compile.rs`, `ops/quant_cache.rs`,
  `kvcache/turboquant.rs`, `prefill/{kv_spill,memory_stage,pflash}.rs`) were **stubs**
  ("Stub for workspace structure") — dropped in favour of live's real implementations.
  No real logic was lost.
- Metal-side improvements (q8 shader, load_vector, race-fix) from `engine-metal-ssd`/
  `engine-metal` do **not** transfer: this is an MLX-C engine (compute via MLX framework,
  not raw Metal). See `CONTEXT.md#cross-engine-notes`.
