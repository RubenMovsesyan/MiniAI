#!/usr/bin/env python3

import os
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
INPUTS_DIR = os.path.join(DATA_DIR, "inputs")
EXPECTED_DIR = os.path.join(DATA_DIR, "expected")

os.makedirs(INPUTS_DIR, exist_ok=True)
os.makedirs(os.path.join(EXPECTED_DIR, "softmax"), exist_ok=True)

# (variant_name, rows, cols, scale) — scale widens the logit range.
# The large-logit case (scale 30) proves max-subtraction prevents inf/NaN.
test_cases = [
    ("2x3_f32", 2, 3, 1.0),
    ("3x4_f32", 3, 4, 1.0),
    ("10x10_f32", 10, 10, 1.0),
    ("100x100_f32", 100, 100, 1.0),
    ("1x100_f32", 1, 100, 1.0),
    ("100x1_f32", 100, 1, 1.0),
    ("8x8_large_f32", 8, 8, 30.0),
]

for variant_name, rows, cols, scale in test_cases:
    variant_dir = os.path.join(INPUTS_DIR, variant_name)
    os.makedirs(variant_dir, exist_ok=True)

    np.random.seed(42 + rows * cols)
    A = (np.random.randn(rows, cols) * scale).astype(np.float32)

    np.savetxt(os.path.join(variant_dir, "A.csv"), A, fmt="%.8e", delimiter=",")

    # Numerically stable softmax reference
    z = A - A.max(axis=1, keepdims=True)
    e = np.exp(z)
    sm = (e / e.sum(axis=1, keepdims=True)).astype(np.float32)

    np.savetxt(
        os.path.join(EXPECTED_DIR, "softmax", f"{variant_name}.csv"),
        sm,
        fmt="%.8e",
        delimiter=","
    )

# Touch sentinel so build.c knows data is up to date
sentinel = os.path.join(DATA_DIR, ".generated")
open(sentinel, "w").close()

print(f"Generated softmax test data for {len(test_cases)} test cases")
