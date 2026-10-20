#!/usr/bin/env bash
# Requantize the GigaChat 3.5 GGUF for heretic-gguf abliteration + inference:
#   * routed MoE experts (the bulk of the parameters) -> Q2_K
#   * everything else (attention, dense FFN, shared experts, SSM, MTP) -> Q6_K
#   * no imatrix (fast, weight-only quantization)
#
# Verified against the same-arch synthetic model (see examples/
# make_synthetic_gigachat35.py); expected automatic fallbacks on the real
# model, all benign:
#   * attn_k_b.weight      -> q8_0   (ne0 = key_mla, not divisible by 256)
#   * ffn_gate_inp.weight  -> f16    (routing gate, kept unquantized by
#                                     llama-quantize's standard logic)
#   * exp_probs_b.bias     -> f32    (8-element bias, K-quants don't fit)
#   * *.norm*.weight       -> f16    (explicit override below)
#   * ssm_conv1d.weight    -> f32    (REQUIRED: the SSM conv op asserts a
#                                     float32 weight in every backend)
# Note: llama-quantize's standard logic forces attn_v-class tensors (attn_v_b,
# attn_qkv, attn_kv_b) to Q8_0 when n_expert == 8 exactly (a Mixtral-era
# rule). The synthetic smoke model trips it; the real model does not.
#
# Usage:  ./requant_gigachat35.sh INPUT.gguf OUTPUT_DIR [nthreads]
# Needs:  llama-quantize built from the pinned llama.cpp (see LLAMA_CPP_PIN),
#         free disk >= size(INPUT) + ~1/3 of it, and RAM for streaming.

set -euo pipefail

IN="${1:?usage: $0 INPUT.gguf OUTPUT_DIR [nthreads]}"
OUTDIR="${2:?usage: $0 INPUT.gguf OUTPUT_DIR [nthreads]}"
NTHREADS="${3:-$(nproc)}"

QUANTIZE="${QUANTIZE:-llama-quantize}"
OUT="$OUTDIR/gigachat35_q6q2"

mkdir -p "$OUTDIR"

echo "input : $IN ($(du -sh "$IN" | cut -f1))"
echo "output: $OUT"
echo "note  : weight-only quantization, no imatrix"
echo

"$QUANTIZE" \
    --allow-requantize \
    --token-embedding-type F16 \
    --output-tensor-type F16 \
    --tensor-type 'blk\..*\.ssm_conv1d\.weight=F32' \
    --tensor-type 'blk\..*ffn_gate_up_exps\.weight=Q2_K' \
    --tensor-type 'blk\..*ffn_down_exps\.weight=Q2_K' \
    --tensor-type 'blk\..*ffn_gate_exps\.weight=Q2_K' \
    --tensor-type 'blk\..*ffn_up_exps\.weight=Q2_K' \
    --tensor-type 'blk\..*ffn_gate_shexp\.weight=Q6_K' \
    --tensor-type 'blk\..*ffn_up_shexp\.weight=Q6_K' \
    --tensor-type 'blk\..*ffn_down_shexp\.weight=Q6_K' \
    --tensor-type '.*norm.*\.weight=F16' \
    "$IN" "$OUT" Q6_K "$NTHREADS" 2>&1 | tee "$OUTDIR/requant.log" \
    | grep -E "^\[|model size|quant size|fallback|not divisible|falling back" || true

echo
echo "=== result ==="
ls -lh "$OUT"*.gguf 2>/dev/null || ls -lh "$OUTDIR"
echo
echo "=== fallbacks (expected: attn_k_b->q8_0 only, plus the small tensors above) ==="
grep -E "not divisible|falling back|F16 due to unusual shape" "$OUTDIR/requant.log" | sort | uniq -c || true
echo
echo "=== type distribution ==="
python3 - "$OUT" <<'EOF'
import collections, glob, sys
from gguf import GGUFReader
files = sorted(glob.glob(sys.argv[1] + "*"))
if not files:
    sys.exit(f"no output found for {sys.argv[1]}*")
dist = collections.Counter()
for f in files:
    r = GGUFReader(f)
    for t in r.tensors:
        dist[t.tensor_type.name] += 1
for name, n in dist.most_common():
    print(f"  {name:8s} {n}")
EOF
echo
echo "done: $OUT*"
