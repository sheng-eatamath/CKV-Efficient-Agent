#!/bin/bash
# ============================================================
# CKV-Agent Workflow: One-Click Full Pipeline Runner
# ============================================================
# Usage:
#   bash run_all.sh           # Run everything (start server + benchmarks)
#   bash run_all.sh --bench a # Run only benchmark A
#   bash run_all.sh --no-server  # Skip server start (already running)
#
# Environment: ssh cbcb29 && conda activate sync
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Defaults
BENCH="all"
START_SERVER=true
MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
PORT=8000
VLLM_PID=""
CONFIG_PATH="configs/default.yaml"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --bench) BENCH="$2"; shift 2 ;;
        --no-server) START_SERVER=false; shift ;;
        --model) MODEL="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Activate conda env
eval "$(conda shell.bash hook)"
conda activate sync

echo "============================================================"
echo "CKV-Agent Workflow Pipeline"
echo "============================================================"
echo "Benchmark: $BENCH"
echo "Model:     $MODEL"
echo "Port:      $PORT"
echo "Server:    $START_SERVER"
echo "Python:    $(which python)"
echo "============================================================"
echo ""

# Create output dirs
mkdir -p logs results results/figures cache traces

mapfile -t REPLAY_CONFIG < <(
python - <<'PY'
import yaml

with open("configs/default.yaml") as f:
    cfg = yaml.safe_load(f)

replay_cfg = cfg.get("bench_d_replay", {})
print(replay_cfg.get("trace_dir", "traces"))
print(replay_cfg.get("num_traces_per_domain", 30))
PY
)
TRACE_DIR="${REPLAY_CONFIG[0]}"
TRACE_COUNT_PER_DOMAIN="${REPLAY_CONFIG[1]}"

# Step 1: Generate traces if needed
if [[ "$BENCH" == "d" || "$BENCH" == "all" ]]; then
    echo "[1] Generating synthetic replay traces..."
    mkdir -p "$TRACE_DIR"
    find "$TRACE_DIR" -maxdepth 1 -type f -name '*.json' -delete
    python scripts/extract_traces.py \
        --output-dir "$TRACE_DIR" \
        --num-per-domain "$TRACE_COUNT_PER_DOMAIN" \
        --seed 42
    echo ""
fi

# Step 2: Start vLLM server
if $START_SERVER; then
    echo "[2] Starting vLLM server..."
    bash scripts/start_vllm.sh "$MODEL" "$PORT" &
    VLLM_PID=$!
    echo "  vLLM PID: $VLLM_PID"

    # Wait for server to be ready
    echo "  Waiting for server..."
    MAX_WAIT=300
    WAITED=0
    while ! curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; do
        sleep 5
        WAITED=$((WAITED + 5))
        if [ $WAITED -ge $MAX_WAIT ]; then
            echo "  ERROR: vLLM server did not start within ${MAX_WAIT}s"
            kill $VLLM_PID 2>/dev/null || true
            exit 1
        fi
        echo "  ... waiting (${WAITED}s)"
    done
    echo "  Server ready!"
    echo ""
fi

# Step 3: Verify server
echo "[3] Verifying server connection..."
curl -s "http://localhost:${PORT}/metrics" | head -5
echo ""

# Step 4: Run benchmarks
echo "[4] Running benchmark(s): $BENCH"
echo "============================================================"

python scripts/run_benchmark.py \
    --bench "$BENCH" \
    --config "$CONFIG_PATH" \
    --model "$MODEL" \
    --port "$PORT" \
    --output-dir results \
    --run-id "bench_${BENCH}_$(date +%Y%m%d_%H%M%S)" \
    2>&1 | tee "cache/bench_${BENCH}_output.txt"

echo ""

# Step 5: Generate plots
echo "[5] Generating plots..."
python scripts/plot_results.py --results-dir results --output-dir results/figures
echo ""

# Step 6: Summary
echo "============================================================"
echo "PIPELINE COMPLETE"
echo "============================================================"
echo "Results:  results/"
echo "Logs:     logs/"
echo "Figures:  results/figures/"
echo "Cache:    cache/"
echo ""

# List output files
echo "Output files:"
ls -la results/*.json 2>/dev/null || echo "  (no JSON results yet)"
ls -la results/figures/*.png 2>/dev/null || echo "  (no figures yet)"
echo ""

# Cleanup: stop vLLM if we started it
if [ -n "$VLLM_PID" ]; then
    echo "Stopping vLLM server (PID $VLLM_PID)..."
    kill $VLLM_PID 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true
fi

echo "Done."
