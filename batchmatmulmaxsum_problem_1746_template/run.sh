#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

OP_NAME="batch_matmul_max_sum_custom"

# ============================================================
# Test parameters
#
# Usage:
#   bash run.sh
#   bash run.sh B M K N
#
# Examples:
#   bash run.sh 1 1 32 1
#   bash run.sh 1 1 32 8
#
# Current version:
#   Layout = NN
#   dtype  = FP16
# ============================================================

if [ "$#" -ne 0 ] && [ "$#" -ne 4 ]; then
    echo "Usage:"
    echo "  bash run.sh"
    echo "  bash run.sh B M K N"
    echo
    echo "Example:"
    echo "  bash run.sh 1 1 32 8"
    exit 1
fi

# Default = original baseline
B="${1:-1}"
M="${2:-1}"
K="${3:-32}"
N="${4:-1}"

# Basic parameter validation
for value in "${B}" "${M}" "${K}" "${N}"; do
    if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: B, M, K and N must all be positive integers."
        exit 1
    fi
done

echo "============================================================"
echo "BatchMatmulMaxSum Test"
echo "B=${B}, M=${M}, K=${K}, N=${N}"
echo "Layout=NN, dtype=FP16"
echo "============================================================"

# ============================================================
# Check CANN environment
# ============================================================

if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    echo "ERROR: ASCEND_HOME_PATH is not set. Please run:"
    echo "  source /usr/local/Ascend/ascend-toolkit/set_env.sh"
    echo "or set ASCEND_HOME_PATH to your CANN toolkit path."
    exit 1
fi

echo "=== [1/4] Set CANN env ==="
source "${ASCEND_HOME_PATH}/set_env.sh"

# ============================================================
# Build
# ============================================================

echo "=== [2/4] Build ==="

rm -rf build
mkdir -p build

cd build

cmake ..
make -j4

cd ..

# ============================================================
# Generate input + golden using SAME B/M/K/N
# ============================================================

echo "=== [3/4] Gen test data ==="

cd build

python3 ../scripts/gen_data.py \
    "${B}" \
    "${M}" \
    "${K}" \
    "${N}"

# ============================================================
# Run kernel + verify
# ============================================================

echo "=== [4/4] Run + Verify ==="

rm -f input/*.bin

cp input/case0/* input/ 2>/dev/null || true
cp output/golden_case0/* output/ 2>/dev/null || true

find output \
    -name '*.bin' \
    ! -name 'golden_*' \
    -delete 2>/dev/null || true

if timeout 120 "./${OP_NAME}" \
    "${B}" \
    "${M}" \
    "${K}" \
    "${N}"
then
    if python3 ../scripts/verify_result.py 0; then
        echo "============================================================"
        echo "=== PASSED ==="
        echo "B=${B}, M=${M}, K=${K}, N=${N}"
        echo "============================================================"
    else
        echo "============================================================"
        echo "=== FAILED: correctness check failed ==="
        echo "B=${B}, M=${M}, K=${K}, N=${N}"
        echo "============================================================"
        exit 1
    fi
else
    echo "============================================================"
    echo "=== FAILED: kernel exited non-zero or timed out ==="
    echo "B=${B}, M=${M}, K=${K}, N=${N}"
    echo "============================================================"
    exit 1
fi