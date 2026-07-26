#!/usr/bin/env python3

import os
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
INPUTS_DIR = os.path.join(DATA_DIR, "inputs")
EXPECTED_DIR = os.path.join(DATA_DIR, "expected")

os.makedirs(INPUTS_DIR, exist_ok=True)
os.makedirs(os.path.join(EXPECTED_DIR, "row_sum"), exist_ok=True)
os.makedirs(os.path.join(EXPECTED_DIR, "col_sum"), exist_ok=True)
os.makedirs(os.path.join(EXPECTED_DIR, "sum"), exist_ok=True)
os.makedirs(os.path.join(EXPECTED_DIR, "row_max"), exist_ok=True)
os.makedirs(os.path.join(EXPECTED_DIR, "col_max"), exist_ok=True)
os.makedirs(os.path.join(EXPECTED_DIR, "max"), exist_ok=True)
os.makedirs(os.path.join(EXPECTED_DIR, "row_argmax"), exist_ok=True)

test_cases = [
    (2, 3),
    (3, 4),
    (10, 10),
    (100, 100),
    (1, 100),   # single row
    (100, 1),   # single col
]

for rows, cols in test_cases:
    variant_name = f"{rows}x{cols}_f32"
    variant_dir = os.path.join(INPUTS_DIR, variant_name)
    os.makedirs(variant_dir, exist_ok=True)

    # Generate random input matrix
    np.random.seed(42 + rows * cols)
    A = np.random.randn(rows, cols).astype(np.float32)

    # Save input
    np.savetxt(os.path.join(variant_dir, "A.csv"), A, fmt="%.8e", delimiter=",")

    # Compute and save expected outputs
    row_sum_result = np.sum(A, axis=1, keepdims=True).astype(np.float32)
    col_sum_result = np.sum(A, axis=0, keepdims=True).astype(np.float32)
    sum_result = np.sum(A).astype(np.float32)

    row_max_result = np.amax(A, axis=1, keepdims=True).astype(np.float32)
    col_max_result = np.amax(A, axis=0, keepdims=True).astype(np.float32)
    max_result = np.amax(A).astype(np.float32)

    np.savetxt(
        os.path.join(EXPECTED_DIR, "row_sum", f"{variant_name}.csv"),
        row_sum_result,
        fmt="%.8e",
        delimiter=","
    )
    np.savetxt(
        os.path.join(EXPECTED_DIR, "col_sum", f"{variant_name}.csv"),
        col_sum_result,
        fmt="%.8e",
        delimiter=","
    )
    np.savetxt(
        os.path.join(EXPECTED_DIR, "sum", f"{variant_name}.csv"),
        np.array([[sum_result]]),
        fmt="%.8e",
        delimiter=","
    )
    np.savetxt(
        os.path.join(EXPECTED_DIR, "row_max", f"{variant_name}.csv"),
        row_max_result,
        fmt="%.8e",
        delimiter=","
    )
    np.savetxt(
        os.path.join(EXPECTED_DIR, "col_max", f"{variant_name}.csv"),
        col_max_result,
        fmt="%.8e",
        delimiter=","
    )
    np.savetxt(
        os.path.join(EXPECTED_DIR, "max", f"{variant_name}.csv"),
        np.array([[max_result]]),
        fmt="%.8e",
        delimiter=","
    )
    # argmax: index of the row max (np.argmax breaks ties on the lowest index, same as our kernel)
    row_argmax_result = np.argmax(A, axis=1, keepdims=True).astype(np.float32)
    np.savetxt(
        os.path.join(EXPECTED_DIR, "row_argmax", f"{variant_name}.csv"),
        row_argmax_result,
        fmt="%.8e",
        delimiter=","
    )

# Touch sentinel so build.c knows data is up to date
sentinel = os.path.join(DATA_DIR, ".generated")
open(sentinel, "w").close()

print(f"Generated test data for {len(test_cases)} test cases")
