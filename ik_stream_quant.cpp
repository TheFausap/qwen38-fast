#include "ggml.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

extern "C" {

size_t ik_row_size(int type_int, int64_t n_per_row) {
    return ggml_row_size((enum ggml_type)type_int, n_per_row);
}

int64_t ik_row_blck_size(int type_int) {
    return ggml_row_blck_size((enum ggml_type)type_int);
}

int ik_quantize_requires_imatrix(int type_int) {
    return ggml_quantize_requires_imatrix((enum ggml_type)type_int) ? 1 : 0;
}

static bool is_cuda_quant_supported(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_KT:
        case GGML_TYPE_IQ2_KT:
        case GGML_TYPE_IQ3_KT:
        case GGML_TYPE_IQ4_KT:
        case GGML_TYPE_IQ2_KS:
        case GGML_TYPE_IQ3_KS:
        case GGML_TYPE_IQ4_KS:
        case GGML_TYPE_IQ5_KS:
            return true;
        default:
            return false;
    }
}

size_t ik_quantize_tensor(
    int type_int,
    const float * f32_data,
    void * out_buf,
    int64_t nslice,
    int64_t nrows,
    int64_t n_per_row,
    const float * imatrix,
    int use_cuda,
    int n_threads
) {
    const enum ggml_type type = (enum ggml_type)type_int;
    ggml_quantize_init(type);
    const size_t row_size = ggml_row_size(type, n_per_row);

    if (type == GGML_TYPE_F32) {
        size_t nb = (size_t)nslice * (size_t)nrows * (size_t)n_per_row * sizeof(float);
        std::memcpy(out_buf, f32_data, nb);
        return nb;
    }

    if (type == GGML_TYPE_F16) {
        size_t total = (size_t)nslice * (size_t)nrows * (size_t)n_per_row;
        ggml_fp32_to_fp16_row(f32_data, (ggml_fp16_t *)out_buf, (int64_t)total);
        return total * sizeof(ggml_fp16_t);
    }

    if (type == GGML_TYPE_BF16) {
        size_t total = (size_t)nslice * (size_t)nrows * (size_t)n_per_row;
        ggml_fp32_to_bf16_row(f32_data, (ggml_bf16_t *)out_buf, (int64_t)total);
        return total * sizeof(ggml_bf16_t);
    }

    if (use_cuda && is_cuda_quant_supported(type)) {
        size_t total_bytes = 0;
        bool all_ok = true;
        for (int64_t s = 0; s < nslice; ++s) {
            const float * src_s = f32_data + s * nrows * n_per_row;
            char *        dst_s = (char *)out_buf + s * nrows * row_size;
            const float * im_s  = imatrix ? (imatrix + s * n_per_row) : nullptr;
            size_t nb = ggml_cuda_quantize(0, type, src_s, dst_s, nrows, n_per_row, 1, im_s);
            if (nb != (size_t)nrows * row_size) {
                all_ok = false;
                break;
            }
            total_bytes += nb;
        }
        if (all_ok) {
            return total_bytes;
        }
        fprintf(stderr, "[ik_stream_quant] CUDA quant fallback to CPU for type=%d\n", type_int);
    }

    if (n_threads < 1) n_threads = 1;
    size_t total_bytes = 0;

    if (nslice > 1 && n_threads > 1) {
        std::atomic<int64_t> next_slice{0};
        std::atomic<size_t>  bytes_acc{0};
        int nt = (int)std::min<int64_t>(n_threads, nslice);
        std::vector<std::thread> workers;
        workers.reserve(nt);
        for (int t = 0; t < nt; ++t) {
            workers.emplace_back([&]() {
                size_t local_b = 0;
                for (;;) {
                    int64_t s = next_slice.fetch_add(1, std::memory_order_relaxed);
                    if (s >= nslice) break;
                    const float * src_s = f32_data + s * nrows * n_per_row;
                    char *        dst_s = (char *)out_buf + s * nrows * row_size;
                    const float * im_s  = imatrix ? (imatrix + s * n_per_row) : nullptr;
                    local_b += ggml_quantize_chunk(type, src_s, dst_s, 0, nrows, n_per_row, im_s, nullptr);
                }
                bytes_acc.fetch_add(local_b, std::memory_order_relaxed);
            });
        }
        for (auto & w : workers) w.join();
        total_bytes = bytes_acc.load();
    } else if (nslice == 1 && n_threads > 1 && nrows >= 64) {
        std::atomic<int64_t> next_row{0};
        std::atomic<size_t>  bytes_acc{0};
        const int64_t raw_chunk = std::max<int64_t>(32, (nrows + n_threads * 4 - 1) / (n_threads * 4));
        const int64_t chunk_rows = ((raw_chunk + 7) / 8) * 8;
        int nt = (int)std::min<int64_t>(n_threads, (nrows + chunk_rows - 1) / chunk_rows);
        std::vector<std::thread> workers;
        workers.reserve(nt);
        for (int t = 0; t < nt; ++t) {
            workers.emplace_back([&]() {
                size_t local_b = 0;
                for (;;) {
                    int64_t r0 = next_row.fetch_add(chunk_rows, std::memory_order_relaxed);
                    if (r0 >= nrows) break;
                    int64_t nr = std::min(chunk_rows, nrows - r0);
                    local_b += ggml_quantize_chunk(type, f32_data, out_buf, r0 * n_per_row, nr, n_per_row, imatrix, nullptr);
                }
                bytes_acc.fetch_add(local_b, std::memory_order_relaxed);
            });
        }
        for (auto & w : workers) w.join();
        total_bytes = bytes_acc.load();
    } else {
        for (int64_t s = 0; s < nslice; ++s) {
            const float * src_s = f32_data + s * nrows * n_per_row;
            char *        dst_s = (char *)out_buf + s * nrows * row_size;
            const float * im_s  = imatrix ? (imatrix + s * n_per_row) : nullptr;
            total_bytes += ggml_quantize_chunk(type, src_s, dst_s, 0, nrows, n_per_row, im_s, nullptr);
        }
    }

    return total_bytes;
}

} // extern "C"
