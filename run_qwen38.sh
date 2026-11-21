#!/usr/bin/env bash
set -euo pipefail

BASE="/home/fausap/qwen38-fast"
BIN_DIR="$BASE/ik_llama.cpp/build/bin"
MODEL="$BASE/models/qwen38-iq2_kt-perm.gguf"
PLE="$BASE/models/qwen38-ple-q8_0.gguf"

# Auto-size frequency-sorted hot-expert RAM prefix from current MemAvailable
# unless IK_HOT_EXPERTS is explicitly overridden.
if [[ -z "${IK_HOT_EXPERTS:-}" ]]; then
    AVAIL_MIB=$(awk '/^MemAvailable:/ {printf "%d", $2/1024}' /proc/meminfo)
    # Leave >= 4.5 GiB headroom so the kernel never thrashes swap
    BUDGET_MIB=$(( AVAIL_MIB > 6000 ? AVAIL_MIB - 4600 : 1500 ))
    HOT=$(( BUDGET_MIB * 512 / 29800 ))
    (( HOT > 440 )) && HOT=440
    (( HOT < 240 )) && HOT=240
    export IK_HOT_EXPERTS="$HOT"
fi
export IK_PLE_MADV_COLD="${IK_PLE_MADV_COLD:-1}"

MODE="${1:-cli}"
shift || true

# Quality / speed profiles:
#   fast   (default): ~12-14 tok/s decode, 14-16 tok/s prefill (blk.0..1 + blk.2..3 down on GPU, -ser 5,0.6)
#   turbo           : ~15+ tok/s decode, 18+ tok/s prefill (blk.0..1 + blk.2..3 down on GPU, -ser 4,0.7)
#   hybrid          : ~11 tok/s decode (20 entry/exit layers 100% exact top-10 + 69/512 static GPU hot-cache, middle 28L -ser 5,0.65)
#   exact           : ~9.2-9.7 tok/s decode (100% exact top-10 across all 48 layers, zero pruning)
#   mtp             : ~11.4 tok/s decode (blk.0..1 + blk.2 down on GPU, -ser 5,0.65, native blk.48 MTP n_max=2)
PROFILE="${PROFILE:-fast}"

COMMON_ARGS=(
    -m "$MODEL"
    --ple-file "$PLE"
    -dple
    -dexp
    -prexp
    -prexp-t "${PREXP_THREADS:-2}"
    -ngl 99
    -ot 'token_embd\.weight=CPU'
    -t "${THREADS:-8}"
    -tb "${THREADS_BATCH:-10}"
    -c "${CTX_SIZE:-8192}"
    -ub "${UBATCH_SIZE:-128}"
    -ctk q8_0
    -ctv q8_0
    -fa on
)

case "$PROFILE" in
    fast)
        COMMON_ARGS+=(
            -ot 'blk\.([2-9]|[1-4][0-9])\.ffn_(gate|up)_exps\.weight=CPU'
            -ot 'blk\.([4-9]|[1-4][0-9])\.ffn_down_exps\.weight=CPU'
            -ser 5,0.6
        )
        ;;
    turbo)
        COMMON_ARGS+=(
            -ot 'blk\.([2-9]|[1-4][0-9])\.ffn_(gate|up)_exps\.weight=CPU'
            -ot 'blk\.([4-9]|[1-4][0-9])\.ffn_down_exps\.weight=CPU'
            -ser 4,0.7
        )
        ;;
    hybrid)
        export IK_GPU_STATIC_HOT_CACHE=1
        COMMON_ARGS+=(
            -ot 'blk\.[0-9]+\.ffn_(gate|up|down)_exps\.weight=CPU'
            --moe-cache-mib "${MOE_CACHE_MIB:-1900}"
            --moe-cache-cpu-share 1.0
            -ser 5,0.65
        )
        ;;
    exact)
        COMMON_ARGS+=(
            -ot 'blk\.([2-9]|[1-4][0-9])\.ffn_(gate|up)_exps\.weight=CPU'
            -ot 'blk\.([4-9]|[1-4][0-9])\.ffn_down_exps\.weight=CPU'
        )
        ;;
    mtp)
        COMMON_ARGS+=(
            -ot 'blk\.([2-9]|[1-4][0-9])\.ffn_(gate|up)_exps\.weight=CPU'
            -ot 'blk\.([3-9]|[1-4][0-9])\.ffn_down_exps\.weight=CPU'
            -ser 5,0.65
            --spec-type "mtp:n_max=${MTP_N:-2},p_min=0.0"
        )
        ;;
    *)
        echo "Unknown PROFILE='$PROFILE' (expected: fast, turbo, hybrid, exact, mtp)" >&2
        exit 1
        ;;
esac

if [[ "$MODE" == "server" ]]; then
    exec "$BIN_DIR/llama-server" "${COMMON_ARGS[@]}" --host 127.0.0.1 --port "${PORT:-8080}" "$@"
elif [[ "$MODE" == "chat" ]]; then
    exec "$BIN_DIR/llama-cli" "${COMMON_ARGS[@]}" -cnv --temp 0.6 --top-p 0.95 "$@"
elif [[ "$MODE" == "bench" ]]; then
    exec "$BIN_DIR/llama-cli" "${COMMON_ARGS[@]}" -n "${N_PREDICT:-256}" --no-display-prompt "$@"
else
    exec "$BIN_DIR/llama-cli" "${COMMON_ARGS[@]}" "$@"
fi
