import numpy as np
import os
import sys

try:
    from ml_dtypes import bfloat16
except ImportError:
    bfloat16 = None

sys.path.insert(0, os.path.dirname(__file__))

from BatchMatmulMaxSum import impl


# ============================================================
# Parse command-line parameters
#
# Usage:
#   python3 gen_data.py
#   python3 gen_data.py B M K N
#
# Current version:
#   Layout = NN
#   dtype  = FP16
# ============================================================

# Original baseline defaults
B = 1
M = 1
K = 32
N = 1

if len(sys.argv) not in (1, 5):
    print(
        f"Usage:\n"
        f"  python3 {sys.argv[0]}\n"
        f"  python3 {sys.argv[0]} B M K N\n\n"
        f"Example:\n"
        f"  python3 {sys.argv[0]} 1 1 32 8"
    )
    sys.exit(1)

if len(sys.argv) == 5:
    try:
        B = int(sys.argv[1])
        M = int(sys.argv[2])
        K = int(sys.argv[3])
        N = int(sys.argv[4])
    except ValueError:
        print("ERROR: B, M, K and N must be integers.")
        sys.exit(1)

if min(B, M, K, N) <= 0:
    print("ERROR: B, M, K and N must all be positive.")
    sys.exit(1)


print("============================================================")
print("Generate BatchMatmulMaxSum test data")
print(f"B={B}, M={M}, K={K}, N={N}")
print("Layout=NN, dtype=FP16")
print("============================================================")


# ============================================================
# Prepare directories
# ============================================================

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)


# ============================================================
# Case 0
#
# NN:
#   X1 = [B, M, K]
#   X2 = [B, K, N]
# ============================================================

np.random.seed(42)

os.makedirs(
    "input/case0",
    exist_ok=True
)

os.makedirs(
    "output/golden_case0",
    exist_ok=True
)


# ------------------------------------------------------------
# Generate X1
# ------------------------------------------------------------

x1 = np.random.uniform(
    low=-1.0,
    high=1.0,
    size=(B, M, K)
).astype(np.float16)

x1.tofile(
    "input/case0/x1.bin"
)


# ------------------------------------------------------------
# Generate X2
# ------------------------------------------------------------

x2 = np.random.uniform(
    low=-1.0,
    high=1.0,
    size=(B, K, N)
).astype(np.float16)

x2.tofile(
    "input/case0/x2.bin"
)


# ------------------------------------------------------------
# Layout
# ------------------------------------------------------------

transposeX1 = False
transposeX2 = False


# ------------------------------------------------------------
# Generate golden output
# ------------------------------------------------------------

golden = impl(
    x1,
    x2,
    transposeX1=transposeX1,
    transposeX2=transposeX2
)

if golden is not None:
    golden.tofile(
        "output/golden_case0/golden_y.bin"
    )

    print(
        f"Golden output generated: "
        f"shape={golden.shape}, dtype={golden.dtype}"
    )
else:
    print("ERROR: impl() returned None.")
    sys.exit(1)


# ------------------------------------------------------------
# Information
# ------------------------------------------------------------

print(f"X1 shape: {x1.shape}")
print(f"X2 shape: {x2.shape}")
print(f"Generated test data and golden output for 1 cases.")