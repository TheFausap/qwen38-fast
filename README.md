# Running 180B Qwen3.8-Flash-Next at 12–15+ tok/s on Consumer Hardware

Run the original unquantized **180B-parameter Qwen3.8-Flash-Next** (`Qwen4ExpForConditionalGeneration`, 360 GB BF16) at **12–15+ tok/s** on modest consumer hardware:

- **CPU:** AMD Ryzen 5 3600 (6C/12T Zen 2, AVX2 + FMA + F16C, no AVX-512, 39.6 GB/s peak DDR4 bandwidth)
- **RAM:** 32 GB DDR4 (~21–23 GiB available with desktop apps running)
- **GPU:** NVIDIA GeForce GTX 1660 (Turing `sm_75`, 6 GB VRAM / ~5.4 GiB free, 192 GB/s)
- **Storage:** 245 GB free on internal NVMe SSD (1.36 GB/s random 1 MB O_DIRECT read, 61 µs 4 KB latency); source 360 GB BF16 weights read from a slow USB 2.0 HDD (40.7 MB/s)

> **100% locally quantized** directly from the original BF16 safetensors weights — no pre-quantized weights downloaded.

---

## Measured End-to-End Benchmarks

Measured on the hardware above with desktop applications consuming ~8–10 GiB RAM (~21–23 GiB free) and ~5.4 GiB free VRAM (`CTX=8192`, `Q8_0` KV cache, Flash Attention enabled):

| Profile (`PROFILE=...`) | Prefill | Decode | Latency | Active Experts & Offload Strategy |
|---|---:|---:|---:|---|
| **`turbo`** | **18.79 tok/s** | **15.05 tok/s** | 66.4 ms/tok | `blk.0..1` all + `blk.2..3 down` on GPU, `-ser 4,0.7` on CPU layers |
| **`fast` (default)** | **14.5–15.9 tok/s** | **12.3–14.0 tok/s** | 71.5–81.1 ms/tok | `blk.0..1` all + `blk.2..3 down` on GPU, `-ser 5,0.6` on CPU layers |
| **`mtp`** | 8.25 tok/s | **11.43 tok/s** | 87.5 ms/tok | `blk.0..1` all + `blk.2 down` on GPU, `-ser 5,0.65`, native `blk.48` MTP (`n_max=2`) |
| **`hybrid`** | **13.95 tok/s** | **10.97 tok/s** | 91.2 ms/tok | **20 entry/exit layers (`blk.0..11, 40..47`) 100% exact top-10** with top `69/512` hot experts pinned in VRAM (`0` runtime PCIe uploads); middle 28 layers `-ser 5,0.65` |
| **`exact`** | 10.2–11.0 tok/s | **9.24–9.65 tok/s** | 103.7–108.2 ms/tok | **100% exact top-10 across all 48 layers** (zero expert pruning) |

### Quality Verification (`PROFILE=fast`, `-ser 5,0.6`)
- **Algorithmic coding:** Generated coherent thinking and O(n log n) Longest Increasing Subsequence logic with index/parent reconstruction (including Python 3.10+ `bisect_left(..., key=...)` nuance) at **11.2–14.0 tok/s**.
- **Math reasoning:** Solved the bat-and-ball + multi-item problem (`ball = $0.05`, `bat = $1.05`, `3 bats + 5 balls = $3.40`) cleanly with proper `</think>` and EOS termination.
- **Technical prose:** Accurate explanation of FlashAttention (HBM vs on-chip SRAM tiling, online softmax, fused kernel) at **12.33 tok/s**.

---

## Architecture Breakdown of Qwen3.8-Flash-Next (180B BF16)

Analyzing all 131 safetensors shards (360 GB BF16) reveals why consumer execution is possible once each component is placed on the right memory tier:

| Component | Parameters | BF16 Size | Runtime Placement | Our Quantized Size |
|---|---:|---:|---|---:|
| **Routed MoE Experts** (48L × 512 exp, top-10 active) | 120.80B | 241.59 GB | Hot prefix in RAM + GPU; cold tail on NVMe | **33.14 GB** |
| **N-gram PLE Table** (`blk.2`, 128 shards `[2.5M, 160]`) | 51.23B | 102.47 GB | Deferred on NVMe SSD (`-dple`, 4 KB lookup = 0.4 ms/tok) | **54.40 GB** (`Q8_0`) |
| **Dense Backbone** (36 GDN + 12 QSA attn + HC + shared exp + `lm_head`) | 4.28B | 8.56 GB | **100% resident in GTX 1660 VRAM** | **2.66 GB** |
| **Token Embeddings** (`embed_tokens`) | 0.64B | 1.28 GB | CPU RAM (1-row lookup) | **0.52 GB** (`Q6_K`) |
| **Native MTP Head** (`blk.48`, QSA + 512-expert MoE) | 2.61B | 5.22 GB | Dense on GPU; experts on CPU/NVMe | **1.19 GB** (`Q3_K`/`Q4_0`) |
| **Vision Tower** (Qwen3-VL ViT) | 0.45B | 0.90 GB | Excluded (text-only build) | — |

---

## Key Engineering Techniques

### 1. Decoupled 51B PLE Table on NVMe SSD (`0 MB` RAM)
The 51.23B-parameter Per-Layer Embedding (PLE) table is a pure sparse lookup (only ~16 rows / <4 KB touched per token at layer 2). We converted it into a separate `Q8_0` GGUF (`qwen38-ple-q8_0.gguf`, 51 GiB) loaded with `--ple-file` + `-dple` and patched `ggml_compute_forward_get_rows_q` (`IK_PLE_MADV_COLD=1`) to immediately advise touched PLE pages `MADV_COLD`. Result: **~0 MB RAM footprint** and **0.40 ms/tok** lookup latency.

### 2. Lossless Frequency-Sorted Expert Permutation
Using activation counts from Unsloth's calibration imatrix (`blk.L.ffn_down_exps.weight.counts`), MoE expert usage is heavily skewed:
- Top **75%** (`384/512`) experts account for **92.41%** of activations
- Top **80%** (`410/512`) experts account for **94.70%** of activations
- Top **85%** (`435/512`) experts account for **96.59%** of activations

In [`stream_convert_trunk.py`](stream_convert_trunk.py), we compute the descending-frequency permutation `perm[L]` for each of the 48 trunk layers (saved in [`models/expert_perm.json`](models/expert_perm.json)) and apply it **identically** to:
1. Router rows (`blk.L.ffn_gate_inp.weight`)
2. Routed expert slices (`blk.L.ffn_{gate,up,down}_exps.weight`)
3. Per-expert imatrix vectors (`in_sum2[e] / counts[e]`)

Because router outputs and expert indices are permuted identically, this transform is **100% mathematically lossless** — while packing the hottest experts contiguously at the front of every layer tensor (`0 .. N-1`) for RAM pinning and isolating the coldest tail (`N .. 511`) contiguously at the end for asynchronous NVMe streaming.

### 3. Hardware-Matched Hybrid Quantization (Zen 2 AVX2 + Turing `sm_75`)
Microbenchmarking `ik_llama.cpp`'s IQK kernels on our Ryzen 5 3600 (Zen 2 AVX2, no AVX-512) uncovered two critical constraints:
- Pure `IQ2_KT` trellis decoding on Zen 2 AVX2 is compute-bound (`74.3 ms` for 45 layers), whereas **`IQ2_K_R4`** (`35.9 ms`) and **`IQ2_XXS_R4`** (`39.3 ms`) are **>2× faster** on Zen 2.
- `ffn_down_exps` has inner dimension `K = 640` (`640 % 256 != 0`), so 256-block `_R4` formats cannot be used on `down_exps` — whereas **`IQ2_KT`** supports `blck = 32` tails on both CUDA `sm_75` and AVX2!

Resulting **38.04 GB** hybrid trunk recipe (streamed directly from the 40 MB/s USB 2.0 HDD with a 1-thread sequential page-cache prefetcher + `POSIX_FADV_DONTNEED` eviction so disk I/O overlaps 100% with quantization, zero 360 GB temp file):
- **`blk.0..11` & `blk.40..47` `gate`/`up`:** `IQ2_K_R4` (2.375 bpw, CUDA + AVX2, high quality on entry/exit layers)
- **`blk.12..39` `gate`/`up`:** `IQ2_XXS_R4` (2.0625 bpw, AVX2, fits more hot experts in 32 GB RAM)
- **`blk.0..47` `down`:** `IQ2_KT` (2.20 bpw, quantized on GTX 1660 via `kt-encoder.cu`)
- **Dense backbone:** `Q6_K` (`output`, `token_embd`), `Q5_K` (`attn_v`, `attn_output`, `ssm_out`, `ple_key/value`), `Q4_K` (other 2D projections), `Q8_0` (`hc_*_up`, `ffn_down_shexp`, `ssm_beta_alpha`), `F32` (norms/biases/conv1d/routers)
- **MTP `blk.48` experts:** `Q3_K` (`gate`/`up`) + `Q4_0` (`down`)

### 4. Surgical `ik_llama.cpp` Engine Patches ([`patches/ik_llama_qwen38.patch`](patches/ik_llama_qwen38.patch))
Applied on top of `ikawrakow/ik_llama.cpp` (`7c00f8788e303bfba2886138e288effb0fd5fbbf`):
1. **Hot-Prefix RAM Pinning & Cold-Tail Isolation (`IK_HOT_EXPERTS`, `IK_PLE_MADV_COLD`):**
   - At load time (`src/llama.cpp`), warms experts `[0..N)` of every CPU-hosted 3D expert tensor into RAM in parallel (`MADV_WILLNEED | MADV_POPULATE_READ`) and marks `[N..512)` as `MADV_RANDOM | MADV_DONTNEED`.
   - At runtime (`ggml/src/ggml.c`, `ggml/src/ggml-moe-prefetch.cpp`), marks touched cold-tail experts (`id >= N`, guarded by a zero-overhead `has_cold` check) and PLE pages `MADV_COLD`, cutting NVMe reads per 256 tokens from **23.13 GiB down to 2.31 GiB (98.48% RAM hit rate)**!
2. **Static Zero-Upload GPU Hot-Expert Cache (`IK_GPU_STATIC_HOT_CACHE=1`):**
   - In `src/llama-moe-cache.cpp`, skips CPU-only `IQ2_XXS_R4` layers so 100% of the `--moe-cache-mib` VRAM budget pins the top `K` frequency-sorted experts (`69/512` per layer in 1,900 MiB) across all 20 CUDA-native layers once at startup. Hits execute at 192 GB/s on the GTX 1660 and misses fall back to CPU with **0 bytes uploaded over PCIe during generation** (~35% GPU hit rate, **+25.7% faster** than static layer offload at identical thermals).
3. **CPU-Safe Smart Expert Reduction (`-ser`):**
   - In `src/llama-build-context.cpp`, re-enabled `ggml_top_k_thresh` selectively on CPU-hosted MoE layers (leaving GPU-offloaded entry layers at exact top-10 and preventing `-1` sentinel crashes in `ggml_cuda_mul_mat_q_id`).
4. **Env-Gated Split Profiler (`IK_SPLIT_PROFILE=1`):**
   - In `ggml/src/ggml-backend.cpp`, zero-overhead-when-unset per-step breakdown of `gpu`, `cpu` (`moe`/`getrows`/`other`), `copy`, and `pref` timings.

---

## Repository Contents

| File | Description |
|---|---|
| [`stream_convert_trunk.py`](stream_convert_trunk.py) | Single-pass streaming converter + lossless frequency-sorted expert permuter + overlapped USB/HDD prefetcher + CUDA/CPU quantizer |
| [`ik_stream_quant.cpp`](ik_stream_quant.cpp) | C++ shared library wrapper (`libik_stream_quant.so`) calling `ik_llama.cpp`'s `ggml_cuda_quantize` (`IQ2_KT`) and multithreaded `ggml_quantize_chunk` |
| [`patches/ik_llama_qwen38.patch`](patches/ik_llama_qwen38.patch) | Complete git diff for `ikawrakow/ik_llama.cpp` (`7c00f878`) |
| [`run_qwen38.sh`](run_qwen38.sh) | Turnkey launcher with auto-sized `IK_HOT_EXPERTS`, 5 speed/quality profiles (`fast`, `turbo`, `hybrid`, `exact`, `mtp`), and `cli`/`chat`/`bench`/`server` modes |
| [`models/expert_perm.json`](models/expert_perm.json) | Per-layer descending-frequency expert permutations (`blk.0..47`) derived from Unsloth's imatrix counts |

---

## Reproduction Guide

### 1. Clone & Build Patched `ik_llama.cpp`
```bash
git clone https://github.com/ikawrakow/ik_llama.cpp.git
cd ik_llama.cpp
git checkout 7c00f8788e303bfba2886138e288effb0fd5fbbf
git apply ../patches/ik_llama_qwen38.patch

cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DGGML_NATIVE=ON \
  -DGGML_IQK_MUL_MAT=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DGGML_OPENMP=ON \
  -DLLAMA_BUILD_TESTS=OFF
cmake --build build -j 8 --target llama-cli llama-server
cd ..
```

### 2. Compile Streaming Quantizer Wrapper
```bash
g++ -O3 -shared -fPIC ik_stream_quant.cpp \
  -I ik_llama.cpp/ggml/include \
  -L ik_llama.cpp/build/ggml/src \
  -Wl,-rpath,$PWD/ik_llama.cpp/build/ggml/src \
  -lggml -lpthread \
  -o libik_stream_quant.so
```

### 3. Set Up Upstream `qwen4exp` Converter & Calibration Stats
```bash
git clone --filter=blob:none --no-checkout https://github.com/ggml-org/llama.cpp.git llama_converter
cd llama_converter
git sparse-checkout set convert_hf_to_gguf.py conversion gguf-py
git checkout 374d90a9fc3b3122473e0879041cff2eeb3d568f
cd ..

# Python environment (numpy, torch, safetensors, transformers, sentencepiece, protobuf, pyyaml, tqdm)
uv venv .venv && uv pip install --python .venv/bin/python numpy torch safetensors transformers sentencepiece protobuf pyyaml tqdm

# Download Unsloth imatrix calibration stats (~554 MB, stats only — not model weights)
mkdir -p models
curl -L -o models/imatrix_unsloth.gguf_file \
  https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/main/imatrix_unsloth.gguf_file
```

### 4. Convert PLE Table + Stream-Quantize Permuted Trunk/MTP
```bash
# Step A: Convert the 51B PLE table once -> models/qwen38-ple-q8_0.gguf (54.4 GB)
.venv/bin/python llama_converter/convert_hf_to_gguf.py /mnt/LLM/MODELS \
  --ple-only --outtype q8_0 --outfile models/qwen38-ple-q8_0.gguf

# Step B: Stream-quantize + frequency-permute trunk & MTP -> models/qwen38-iq2_kt-perm.gguf (38.04 GB)
.venv/bin/python stream_convert_trunk.py --model-dir /mnt/LLM/MODELS --threads 10
```

### 5. Run Inference
```bash
# Interactive chat (default PROFILE=fast, ~12–14 tok/s)
./run_qwen38.sh chat

# OpenAI-compatible API + Web UI on http://127.0.0.1:8080
./run_qwen38.sh server

# Choose a profile:
PROFILE=turbo  ./run_qwen38.sh chat   # ~15+ tok/s (-ser 4,0.7)
PROFILE=hybrid ./run_qwen38.sh chat   # ~11 tok/s (20L exact + static GPU hot cache)
PROFILE=exact  ./run_qwen38.sh chat   # ~9.2–9.7 tok/s (100% exact top-10)
PROFILE=mtp    ./run_qwen38.sh chat   # ~11.4 tok/s (native blk.48 MTP)
```
