#!/bin/bash
# ============================================================
# CKV-Agent Workflow: One-Click Test Runner
# ============================================================
# Usage: bash run_tests.sh
#
# This script runs ALL offline tests (no vLLM server needed).
# Tests use a mock backend.
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "CKV-Agent Workflow: Running Tests"
echo "============================================================"
echo ""

# Activate conda env
eval "$(conda shell.bash hook)"
conda activate sync

echo "Python: $(which python)"
echo "Working dir: $(pwd)"
echo ""

mkdir -p cache

# Step 1: Generate synthetic traces (needed for replay tests)
echo "[1/3] Generating synthetic traces..."
python scripts/extract_traces.py --output-dir traces --num-per-domain 5 --seed 42
echo ""

# Step 2: Run unit tests
echo "[2/3] Running unit tests..."
python -m pytest tests/test_ckv_workflow.py -v --tb=short 2>&1 | tee cache/test_results.txt
echo ""

# Step 3: Quick import validation
echo "[3/3] Validating imports..."
python -c "
from src_workflow.events import EventBus, WorkflowEvent, EventType
from src_workflow.vllm_backend import VLLMBackend, GenerateResult
from src_workflow.agent_step import run_step
from src_workflow.tools import TOOL_REGISTRY
from src_workflow.profiler import KVProfiler
from src_workflow.workflows.retry import RetryWorkflow
from src_workflow.workflows.stall import StallWorkflow
from src_workflow.workflows.branch import BranchWorkflow
from src_workflow.workflows.replay import ReplayWorkflow, ReplayPlan
print('All imports OK')
"
echo ""

echo "============================================================"
echo "All tests passed!"
echo "============================================================"
