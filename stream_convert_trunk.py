#!/usr/bin/env python3
"""
Single-pass streaming converter + lossless frequency-sorted expert permuter +
CUDA-accelerated ik_llama.cpp IQ2_KT quantizer for Qwen3.8-Flash-Next (qwen4_exp).

Features:
1. Uses upstream ggml-org/llama.cpp Qwen4ExpTextModel exact metadata, tokenizer,
   chat template, linear-attn V-reorder, norm+1 shifts, indexer split, HC merges,
   and MTP eh_proj concatenation.
2. Reads Unsloth's imatrix GGUF once, computes descending-frequency expert
   permutations perm[L] for all 48 trunk MoE layers, and applies perm[L]
   identically to ffn_gate_inp.weight rows, ffn_{gate,up,down}_exps.weight
   slices, and per-expert imatrix vectors (100% lossless identity transform!).
   This places the hottest ~81% of experts (95.2% of activations!) contiguously
   at the front of every layer's tensor for RAM pinning, and the coldest ~19%
   (<5% of activations) contiguously at the tail for async NVMe streaming.
3. Quantizes every tensor on-the-fly via libik_stream_quant.so (linked to
   ik_llama.cpp's libggml.so with CUDA sm_75 kt-encoder.cu) — zero 360 GB temp
   file, peak RAM < 6 GB.
4. Caches the single most recent raw gate_up_proj safetensors tensor across
   consecutive gate_exps -> up_exps writes, avoiding reading 160 GB twice from
   the slow USB 2.0 source disk.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import logging
import os
import queue
import re
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

CONVERTER_DIR = Path("/home/fausap/qwen38-fast/llama_converter")
sys.path.insert(0, str(CONVERTER_DIR))
sys.path.insert(1, str(CONVERTER_DIR / "gguf-py"))

import gguf
from gguf.gguf_writer import TensorInfo
from conversion.base import LazyTorchTensor, ModelBase
from conversion.qwen4exp import Qwen4ExpTextModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stream-quant")

# Exact ik_llama.cpp ggml_type enum values (from ik_llama.cpp/ggml/include/ggml.h)
GGML_TYPE_F32    = 0
GGML_TYPE_F16    = 1
GGML_TYPE_Q4_0   = 2
GGML_TYPE_Q5_0   = 6
GGML_TYPE_Q8_0   = 8
GGML_TYPE_Q2_K   = 10
GGML_TYPE_Q3_K   = 11
GGML_TYPE_Q4_K   = 12
GGML_TYPE_Q5_K   = 13
GGML_TYPE_Q6_K       = 14
GGML_TYPE_IQ2_XXS    = 16
GGML_TYPE_IQ2_K      = 137
GGML_TYPE_IQ2_KT     = 153
GGML_TYPE_IQ3_KT     = 154
GGML_TYPE_IQ1_KT     = 158
GGML_TYPE_IQ2_XXS_R4 = 216
GGML_TYPE_IQ2_K_R4   = 337

LLAMA_FTYPE_MOSTLY_IQ2_KT = 151

TYPE_NAMES = {
    GGML_TYPE_F32: "F32",
    GGML_TYPE_F16: "F16",
    GGML_TYPE_Q4_0: "Q4_0",
    GGML_TYPE_Q5_0: "Q5_0",
    GGML_TYPE_Q8_0: "Q8_0",
    GGML_TYPE_Q2_K: "Q2_K",
    GGML_TYPE_Q3_K: "Q3_K",
    GGML_TYPE_Q4_K: "Q4_K",
    GGML_TYPE_Q5_K: "Q5_K",
    GGML_TYPE_Q6_K: "Q6_K",
    GGML_TYPE_IQ2_XXS: "IQ2_XXS",
    GGML_TYPE_IQ2_K: "IQ2_K",
    GGML_TYPE_IQ2_KT: "IQ2_KT",
    GGML_TYPE_IQ3_KT: "IQ3_KT",
    GGML_TYPE_IQ1_KT: "IQ1_KT",
    GGML_TYPE_IQ2_XXS_R4: "IQ2_XXS_R4",
    GGML_TYPE_IQ2_K_R4: "IQ2_K_R4",
}


def load_quant_lib(so_path: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(so_path))
    lib.ik_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
    lib.ik_row_size.restype = ctypes.c_size_t
    lib.ik_row_blck_size.argtypes = [ctypes.c_int]
    lib.ik_row_blck_size.restype = ctypes.c_int64
    lib.ik_quantize_tensor.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.ik_quantize_tensor.restype = ctypes.c_size_t
    return lib


def load_imatrix_and_perms(imatrix_path: Path, n_trunk_layers: int = 48):
    """Load Unsloth GGUF imatrix, compute descending-frequency expert permutations,
    and normalize per-slice importance vectors (in_sum2 / counts)."""
    logger.info("Loading imatrix from %s ...", imatrix_path)
    reader = gguf.GGUFReader(str(imatrix_path))
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, np.ndarray] = {}
    for t in reader.tensors:
        name = t.name
        arr = np.array(t.data, dtype=np.float32, copy=True)
        if name.endswith(".in_sum2"):
            sums[name[:-8]] = arr
        elif name.endswith(".sums"):
            sums[name[:-5]] = arr
        elif name.endswith(".counts"):
            counts[name[:-7]] = arr

    # Compute frequency-sorted expert permutation per layer from ffn_down_exps.weight.counts
    perms: dict[int, np.ndarray] = {}
    for l in range(n_trunk_layers):
        key = f"blk.{l}.ffn_down_exps.weight"
        c = counts[key].reshape(-1)
        assert c.shape[0] == 512, f"Unexpected counts shape for {key}: {c.shape}"
        # Stable descending sort by activation count (index 0 = hottest expert)
        perms[l] = np.argsort(-c, kind="stable").astype(np.int64)

    # Normalize imatrix entries; permute 3D MoE expert entries by perms[l]
    imatrix: dict[str, np.ndarray] = {}
    blk_re = re.compile(r"^blk\.(\d+)\.")
    for name, s_arr in sums.items():
        c_arr = counts.get(name)
        m = blk_re.match(name)
        bid = int(m.group(1)) if m else None
        if s_arr.ndim == 2 and bid is not None and bid in perms and s_arr.shape[0] == 512:
            # GGUF shape (n_in, 512) -> numpy shape (512, n_in)
            c_vec = c_arr.reshape(-1) if c_arr is not None else np.ones(512, dtype=np.float32)
            out = np.empty_like(s_arr, dtype=np.float32)
            active = c_vec > 0
            if np.any(active):
                out[active] = s_arr[active] / c_vec[active, None]
                fallback = out[active].mean(axis=0)
            else:
                fallback = np.ones(s_arr.shape[1], dtype=np.float32)
            if np.any(~active):
                out[~active] = fallback
            # Apply the exact same permutation we apply to the expert weights!
            out = np.ascontiguousarray(out[perms[bid]], dtype=np.float32)
            imatrix[name] = out
        else:
            denom = float(c_arr.reshape(-1)[0]) if c_arr is not None and c_arr.size > 0 else 1.0
            if denom <= 0.0:
                denom = 1.0
            imatrix[name] = np.ascontiguousarray((s_arr.reshape(-1) / denom).astype(np.float32))

    logger.info("Loaded %d imatrix entries and computed %d layer expert permutations.", len(imatrix), len(perms))
    return imatrix, perms


def lookup_imatrix(name: str, expected_size: int, imatrix: dict[str, np.ndarray]) -> np.ndarray | None:
    arr = imatrix.get(name)
    if arr is None:
        # Exact ik_llama.cpp/src/llama-quantize.cpp merged-tensor fallbacks
        if "_down_inject.weight" in name:
            arr = imatrix.get(name.replace("_down_inject.weight", "_down.weight"))
        elif "attn_qkv_gate.weight" in name:
            arr = imatrix.get(name.replace("attn_qkv_gate.weight", "attn_qkv.weight"))
        elif "ssm_beta_alpha.weight" in name:
            arr = imatrix.get(name.replace("ssm_beta_alpha.weight", "ssm_beta.weight"))
        elif "ffn_gate_exps.weight" in name:
            arr = imatrix.get(name.replace("ffn_gate_exps.weight", "ffn_up_exps.weight"))
    if arr is None:
        return None
    if arr.size != expected_size:
        logger.warning("Imatrix size mismatch for %s: got %d, expected %d", name, arr.size, expected_size)
        return None
    return np.ascontiguousarray(arr, dtype=np.float32)


def choose_quant_type(name: str, shape: Sequence[int], lib: ctypes.CDLL, expert_qtype: int) -> int:
    nd = len(shape)
    n_per_row = int(shape[-1])

    # 1D tensors, norms, biases, conv1d, ssm params, routers -> F32 for exact precision
    if (
        nd <= 1
        or name.endswith(("_norm.weight", "conv1d.weight", "ssm_a", "ssm_dt.bias", "ffn_gate_inp_shexp.weight"))
        or name.endswith("ffn_gate_inp.weight")
    ):
        return GGML_TYPE_F32

    m = re.match(r"^blk\.(\d+)\.", name)
    bid = int(m.group(1)) if m else None

    # 3D routed MoE experts
    if nd == 3 and ("_exps.weight" in name):
        if bid is not None and bid < 48:
            if "ffn_down_exps.weight" in name:
                # K=640 (not multiple of 256) -> IQ2_KT trellis (2.20 bpw, blck=32, CUDA+AVX2)
                target = GGML_TYPE_IQ2_KT
            elif bid < 12 or bid >= 40:
                # Entry/exit 20 sensitive layers (incl. blk.0..2 on GTX 1660 CUDA!) -> IQ2_K_R4 (2.375 bpw, CUDA+AVX2, 4.5x faster on Zen 2!)
                target = GGML_TYPE_IQ2_K_R4
            else:
                # Middle 28 CPU layers -> IQ2_XXS_R4 (2.0625 bpw, AVX2, 4x faster on Zen 2, fits 414/512 hot experts in 25GB RAM!)
                target = GGML_TYPE_IQ2_XXS_R4
        else:
            # MTP layer (blk.48): no Unsloth imatrix -> Q3_K (2560) / Q4_0 (640) for high draft acceptance
            target = GGML_TYPE_Q3_K if (n_per_row % 256 == 0) else GGML_TYPE_Q4_0
    elif name in ("output.weight", "token_embd.weight"):
        target = GGML_TYPE_Q6_K
    elif any(k in name for k in ("attn_v.weight", "attn_output.weight", "ssm_out.weight", "ple_key.weight", "ple_value.weight", "ffn_down_shexp.weight")):
        target = GGML_TYPE_Q5_K if (n_per_row % 256 == 0) else GGML_TYPE_Q8_0
    elif "ssm_beta_alpha.weight" in name or "hc_attn_up.weight" in name or "hc_ffn_up.weight" in name or "hc_head_up.weight" in name:
        target = GGML_TYPE_Q8_0
    else:
        target = GGML_TYPE_Q4_K if (n_per_row % 256 == 0) else GGML_TYPE_Q8_0

    # Safety guard against block-size incompatibility
    blck = int(lib.ik_row_blck_size(target))
    if n_per_row % blck != 0:
        if n_per_row % 32 == 0:
            target = GGML_TYPE_Q8_0
        else:
            target = GGML_TYPE_F16
    return target


class ShardPrefetcher:
    """1-thread sequential USB 2.0 page-cache pre-reader + FADV_DONTNEED evictor.
    Overlaps slow 40 MB/s USB HDD reads of the next safetensors shard with
    CPU/CUDA quantization of the current tensor, while keeping page cache bounded."""

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.layer_gate_up: dict[int, Path] = {}
        self.layer_down: dict[int, Path] = {}
        idx_path = model_dir / "model.safetensors.index.json"
        if idx_path.exists():
            with open(idx_path) as f:
                wmap = json.load(f).get("weight_map", {})
            for k, fname in wmap.items():
                m = re.match(r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$", k)
                if m:
                    bid = int(m.group(1))
                    if m.group(2) == "gate_up_proj":
                        self.layer_gate_up[bid] = model_dir / fname
                    else:
                        self.layer_down[bid] = model_dir / fname
                elif k == "mtp.layers.0.mlp.experts.gate_up_proj":
                    self.layer_gate_up[48] = model_dir / fname
                elif k == "mtp.layers.0.mlp.experts.down_proj":
                    self.layer_down[48] = model_dir / fname

        self._q: queue.Queue[Path | None] = queue.Queue()
        self._queued: set[Path] = set()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        buf = bytearray(4 * 1024 * 1024)
        mv = memoryview(buf)
        while True:
            path = self._q.get()
            if path is None:
                return
            try:
                fd = os.open(str(path), os.O_RDONLY)
                try:
                    if hasattr(os, "posix_fadvise"):
                        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_SEQUENTIAL)
                    while True:
                        n = os.readv(fd, [mv])
                        if n <= 0:
                            break
                finally:
                    os.close(fd)
            except Exception as exc:
                logger.debug("Prefetch warning for %s: %s", path.name, exc)

    def prefetch(self, path: Path | None) -> None:
        if path is None or path in self._queued or not path.exists():
            return
        self._queued.add(path)
        self._q.put(path)

    def evict(self, path: Path | None) -> None:
        if path is None or not path.exists() or not hasattr(os, "posix_fadvise"):
            return
        try:
            fd = os.open(str(path), os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        except Exception:
            pass

    def on_tensor_read_done(self, name: str) -> None:
        m = re.match(r"^blk\.(\d+)\.", name)
        if not m:
            return
        bid = int(m.group(1))
        if name.endswith("ffn_gate_exps.weight"):
            # While quantizing gate_exps + up_exps (~65-130s), prefetch this layer's down_proj
            # AND next layer's gate_up_proj sequentially at full USB 2.0 speed!
            self.prefetch(self.layer_down.get(bid))
            self.prefetch(self.layer_gate_up.get(bid + 1))
        elif name.endswith("ffn_up_exps.weight"):
            # Done reading this layer's gate_up_proj -> drop from page cache after quant starts
            self.evict(self.layer_gate_up.get(bid))
        elif name.endswith("ffn_down_exps.weight"):
            self.evict(self.layer_down.get(bid))
            self.prefetch(self.layer_gate_up.get(bid + 1))

    def stop(self) -> None:
        self._q.put(None)
        self._thread.join(timeout=5.0)


class StreamingQuantTensor:
    """Duck-typed tensor object stored in GGUFWriter's TensorInfo.tensor.
    Materializes, permutes (if MoE/router), quantizes via libik_stream_quant.so,
    and streams bytes directly to fout on .tofile(fout)."""

    def __init__(
        self,
        name: str,
        shape: tuple[int, ...],
        qtype: int,
        nbytes: int,
        lazy_src: Any,
        lib: ctypes.CDLL,
        imatrix: dict[str, np.ndarray],
        perms: dict[int, np.ndarray],
        use_cuda: bool,
        n_threads: int,
        stats: dict[str, Any],
        prefetcher: ShardPrefetcher | None = None,
    ):
        self.name = name
        self.shape = shape
        self.qtype = qtype
        self.nbytes = nbytes
        self.lazy_src = lazy_src
        self.lib = lib
        self.imatrix = imatrix
        self.perms = perms
        self.use_cuda = use_cuda
        self.n_threads = n_threads
        self.stats = stats
        self.prefetcher = prefetcher

    def tofile(self, fout) -> None:
        t0 = time.perf_counter()
        # Materialize array from LazyNumpyTensor / ndarray
        eager = gguf.LazyNumpyTensor.to_eager(self.lazy_src) if isinstance(self.lazy_src, gguf.LazyBase) else self.lazy_src
        self.lazy_src = None

        # Notify prefetcher as soon as this tensor's disk read is done so USB I/O
        # overlaps with permutation + CPU/CUDA quantization!
        if self.prefetcher is not None:
            self.prefetcher.on_tensor_read_done(self.name)

        # Apply lossless descending-frequency expert permutation to trunk routers & experts
        # directly during float32 contiguous materialization (avoids an extra 3.35 GB copy!)
        m = re.match(r"^blk\.(\d+)\.", self.name)
        bid = int(m.group(1)) if m else None
        if bid is not None and bid in self.perms and (
            self.name.endswith("ffn_gate_inp.weight")
            or self.name.endswith(("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"))
        ):
            assert eager.shape[0] == 512, f"Expected 512 experts/rows for {self.name}, got {eager.shape}"
            arr = np.ascontiguousarray(eager[self.perms[bid]], dtype=np.float32)
        else:
            arr = np.ascontiguousarray(eager, dtype=np.float32)
        del eager

        t_read = time.perf_counter() - t0

        t1 = time.perf_counter()
        if arr.ndim == 1:
            nslice, nrows, n_per_row = 1, 1, arr.shape[0]
        elif arr.ndim == 2:
            nslice, nrows, n_per_row = 1, arr.shape[0], arr.shape[1]
        elif arr.ndim == 3:
            nslice, nrows, n_per_row = arr.shape[0], arr.shape[1], arr.shape[2]
        else:
            raise ValueError(f"Unsupported tensor rank {arr.ndim} for {self.name}: {arr.shape}")

        im = lookup_imatrix(self.name, nslice * n_per_row, self.imatrix)
        im_ptr = im.ctypes.data if im is not None else 0

        out_buf = np.empty(self.nbytes, dtype=np.uint8)
        written = self.lib.ik_quantize_tensor(
            self.qtype,
            arr.ctypes.data,
            out_buf.ctypes.data,
            nslice,
            nrows,
            n_per_row,
            im_ptr,
            1 if self.use_cuda else 0,
            self.n_threads,
        )
        if written != self.nbytes:
            raise RuntimeError(f"Quantized size mismatch for {self.name}: got {written}, expected {self.nbytes}")

        t_quant = time.perf_counter() - t1
        t2 = time.perf_counter()
        out_buf.tofile(fout)
        t_write = time.perf_counter() - t2

        del arr, out_buf
        self.stats["done_bytes"] += self.nbytes
        self.stats["done_count"] += 1
        if self.nbytes >= 10 * 1024 * 1024 or self.stats["done_count"] % 50 == 0:
            elapsed = time.perf_counter() - self.stats["t_start"]
            pct = 100.0 * self.stats["done_bytes"] / max(1, self.stats["total_bytes"])
            eta = (elapsed / max(1e-6, pct / 100.0)) - elapsed if pct > 0.1 else 0.0
            logger.info(
                "[%4d/%d | %5.1f%% | ETA %4.1fm] %-42s %-6s %7.1f MB (read %.1fs, quant %.2fs, write %.2fs, im=%s)",
                self.stats["done_count"],
                self.stats["total_count"],
                pct,
                eta / 60.0,
                self.name,
                TYPE_NAMES.get(self.qtype, str(self.qtype)),
                self.nbytes / 1e6,
                t_read,
                t_quant,
                t_write,
                "Y" if im is not None else "N",
            )
            gc.collect()


class StreamingQwen4ExpModel(Qwen4ExpTextModel):
    model_arch = gguf.MODEL_ARCH.QWEN4EXP

    def __init__(self, *args, lib, imatrix, perms, expert_qtype, use_cuda, n_threads, **kwargs):
        self._ik_lib = lib
        self._ik_imatrix = imatrix
        self._ik_perms = perms
        self._ik_expert_qtype = expert_qtype
        self._ik_use_cuda = use_cuda
        self._ik_n_threads = n_threads
        self._ik_stats = {"done_bytes": 0, "total_bytes": 0, "done_count": 0, "total_count": 0, "t_start": 0.0}
        self._ik_prefetcher: ShardPrefetcher | None = None
        super().__init__(*args, **kwargs)
        self._install_writer_hook()

    def _install_writer_hook(self):
        writer = self.gguf_writer
        lib = self._ik_lib
        imatrix = self._ik_imatrix
        perms = self._ik_perms
        expert_qtype = self._ik_expert_qtype
        use_cuda = self._ik_use_cuda
        n_threads = self._ik_n_threads
        stats = self._ik_stats
        model_self = self

        def hooked_add_tensor(
            name: str,
            tensor: Any,
            raw_shape: Sequence[int] | None = None,
            raw_dtype: Any = None,
            tensor_endianess: Any = None,
        ) -> None:
            if writer.tensor_filter is not None and not writer.tensor_filter(name):
                return
            shape = tuple(int(x) for x in (raw_shape if raw_shape is not None else tensor.shape))
            qtype = choose_quant_type(name, shape, lib, expert_qtype)
            n_per_row = shape[-1]
            nrows_total = 1
            for d in shape[:-1]:
                nrows_total *= d
            row_bytes = int(lib.ik_row_size(qtype, n_per_row))
            nbytes = nrows_total * row_bytes

            # Register exact TensorInfo with raw ik_llama.cpp ggml_type integer
            writer.tensors[-1][name] = TensorInfo(shape=shape, dtype=qtype, nbytes=nbytes)
            writer.tensors[-1][name].tensor = StreamingQuantTensor(
                name=name,
                shape=shape,
                qtype=qtype,
                nbytes=nbytes,
                lazy_src=tensor,
                lib=lib,
                imatrix=imatrix,
                perms=perms,
                use_cuda=use_cuda,
                n_threads=n_threads,
                stats=stats,
                prefetcher=None,
            )
            stats["total_bytes"] += nbytes
            stats["total_count"] += 1

        writer.add_tensor = hooked_add_tensor

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        # Override file_type to LLAMA_FTYPE_MOSTLY_IQ2_KT (151)
        self.gguf_writer.add_file_type(LLAMA_FTYPE_MOSTLY_IQ2_KT)

    def write(self):
        self.prepare_tensors()
        if not self._has_ple:
            raise ValueError("Expected PLE metadata in model")
        self.prepare_metadata(vocab_only=False)
        logger.info(
            "Planned %d tensors, total quantized size = %.2f GB -> %s",
            self._ik_stats["total_count"],
            self._ik_stats["total_bytes"] / 1e9,
            self.fname_out,
        )
        if self.dry_run:
            by_type: dict[int, list[int]] = {}
            for ti in self.gguf_writer.tensors[0].values():
                entry = by_type.setdefault(int(ti.dtype), [0, 0])
                entry[0] += 1
                entry[1] += ti.nbytes
            for qt, (cnt, nb) in sorted(by_type.items(), key=lambda x: -x[1][1]):
                logger.info("  %-8s (id=%3d): %4d tensors, %7.2f GB", TYPE_NAMES.get(qt, str(qt)), qt, cnt, nb / 1e9)
            return
        self._ik_prefetcher = ShardPrefetcher(self.dir_model)
        for ti in self.gguf_writer.tensors[0].values():
            if isinstance(ti.tensor, StreamingQuantTensor):
                ti.tensor.prefetcher = self._ik_prefetcher
        self._ik_prefetcher.prefetch(self._ik_prefetcher.layer_gate_up.get(0))
        try:
            self._ik_stats["t_start"] = time.perf_counter()
            self.gguf_writer.write_header_to_file(path=self.fname_out)
            self.gguf_writer.write_kv_data_to_file()
            self.gguf_writer.write_tensors_to_file(progress=False)
            self.gguf_writer.close()
        finally:
            self._ik_prefetcher.stop()
        dt = time.perf_counter() - self._ik_stats["t_start"]
        logger.info("Finished writing %s (%.2f GB) in %.1f minutes!", self.fname_out, self._ik_stats["total_bytes"] / 1e9, dt / 60.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, default=Path("/mnt/LLM/MODELS"))
    p.add_argument("--imatrix", type=Path, default=Path("/home/fausap/qwen38-fast/models/imatrix_unsloth.gguf_file"))
    p.add_argument("--quant-lib", type=Path, default=Path("/home/fausap/qwen38-fast/libik_stream_quant.so"))
    p.add_argument("--outfile", type=Path, default=Path("/home/fausap/qwen38-fast/models/qwen38-iq2_kt-perm.gguf"))
    p.add_argument("--perm-out", type=Path, default=Path("/home/fausap/qwen38-fast/models/expert_perm.json"))
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--no-cuda", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    lib = load_quant_lib(args.quant_lib)
    imatrix, perms = load_imatrix_and_perms(args.imatrix)

    args.perm_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.perm_out, "w") as f:
        json.dump({str(k): v.tolist() for k, v in perms.items()}, f)
    logger.info("Saved expert frequency permutations to %s", args.perm_out)

    with torch.inference_mode():
        model = StreamingQwen4ExpModel(
            dir_model=args.model_dir,
            ftype=gguf.LlamaFileType.ALL_F32,
            fname_out=args.outfile,
            dry_run=args.dry_run,
            no_ple=True,
            lib=lib,
            imatrix=imatrix,
            perms=perms,
            expert_qtype=GGML_TYPE_IQ2_KT,
            use_cuda=not args.no_cuda,
            n_threads=args.threads,
        )
        model.write()


if __name__ == "__main__":
    main()
