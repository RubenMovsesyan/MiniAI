#!/usr/bin/env python3
"""
Generate CSV test fixtures for matrix operation tests.

Run from project root:
    python3 src/matrix/tests/gen_test_data.py

Output layout:
    src/matrix/tests/data/
      inputs/
        {rows}x{cols}_{type}/
          A.csv, B.csv, col.csv, row.csv
      expected/
        {op}/
          {rows}x{cols}_{type}.csv
      .generated   (sentinel — build.c skips re-generation when this is newer than this script)
"""

import numpy as np
import os
import sys

DATA_ROOT = "src/matrix/tests/data"

SMALL_MEDIUM_SIZES = [
    (3, 3), (3, 5), (4, 4), (4, 7), (8, 8),
    (16, 16), (16, 32), (64, 64), (128, 128), (128, 256),
]
LARGE_SIZES = [(512, 512), (1024, 1024)]

TYPES = ["rand_f32", "struct_f32", "rand_i32", "struct_i32"]

SCALARS = [("2.0", 2.0), ("-1.5", -1.5), ("0.5", 0.5)]


def save(path: str, m: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savetxt(path, m, delimiter=",", fmt="%.8e")


def inputs_dir(name: str) -> str:
    return f"{DATA_ROOT}/inputs/{name}"


def expected_path(op: str, name: str) -> str:
    return f"{DATA_ROOT}/expected/{op}/{name}.csv"


# ─── Input generators ─────────────────────────────────────────────────────────

def make_A(rows: int, cols: int, suffix: str) -> np.ndarray:
    if suffix == "rand_f32":
        np.random.seed(42)
        return np.random.uniform(-10.0, 10.0, (rows, cols)).astype(np.float32)
    if suffix == "struct_f32":
        ri = np.arange(1, rows + 1, dtype=np.float32).reshape(-1, 1)
        ci = np.arange(1, cols + 1, dtype=np.float32).reshape(1, -1)
        return (ri * np.float32(0.5) + ci * np.float32(0.3))
    if suffix == "rand_i32":
        np.random.seed(123)
        return np.random.randint(-100, 101, (rows, cols)).astype(np.float32)
    # struct_i32: A[i,j] = i*cols + j + 1
    ri = np.arange(rows, dtype=np.float32).reshape(-1, 1)
    ci = np.arange(cols, dtype=np.float32).reshape(1, -1)
    return (ri * float(cols) + ci + 1.0).astype(np.float32)


def make_B(rows: int, cols: int, suffix: str) -> np.ndarray:
    if suffix == "rand_f32":
        np.random.seed(99)
        return np.random.uniform(-10.0, 10.0, (rows, cols)).astype(np.float32)
    if suffix == "struct_f32":
        ri = np.arange(1, rows + 1, dtype=np.float32).reshape(-1, 1)
        ci = np.arange(1, cols + 1, dtype=np.float32).reshape(1, -1)
        return (ri * np.float32(0.7) + ci * np.float32(0.2))
    if suffix == "rand_i32":
        np.random.seed(777)
        return np.random.randint(-100, 101, (rows, cols)).astype(np.float32)
    # struct_i32: B[i,j] = (rows-i)*cols + (cols-j)
    ri = np.arange(rows, 0, -1, dtype=np.float32).reshape(-1, 1)
    ci = np.arange(cols, 0, -1, dtype=np.float32).reshape(1, -1)
    return (ri * float(cols) + ci).astype(np.float32)


def make_col(rows: int, cols: int) -> np.ndarray:
    np.random.seed(42 + rows + cols)
    return np.random.uniform(-5.0, 5.0, (rows, 1)).astype(np.float32)


def make_row(rows: int, cols: int) -> np.ndarray:
    np.random.seed(99 + rows + cols)
    return np.random.uniform(-5.0, 5.0, (1, cols)).astype(np.float32)


# ─── Per-variant generation ────────────────────────────────────────────────────

def gen_variant(rows: int, cols: int, suffix: str) -> None:
    name = f"{rows}x{cols}_{suffix}"
    d = inputs_dir(name)

    A   = make_A(rows, cols, suffix)
    B   = make_B(rows, cols, suffix)
    col = make_col(rows, cols)
    row = make_row(rows, cols)

    save(f"{d}/A.csv",   A)
    save(f"{d}/B.csv",   B)
    save(f"{d}/col.csv", col)
    save(f"{d}/row.csv", row)

    save(expected_path("add",       name), (A + B).astype(np.float32))
    save(expected_path("sub",       name), (A - B).astype(np.float32))
    save(expected_path("hadamard",  name), (A * B).astype(np.float32))
    save(expected_path("transpose", name), np.ascontiguousarray(A.T).astype(np.float32))

    for (label, s) in SCALARS:
        save(expected_path(f"scalar_x{label}", name), (A * np.float32(s)).astype(np.float32))

    save(expected_path("colAdd", name), (A + col).astype(np.float32))
    save(expected_path("rowAdd", name), (A + row).astype(np.float32))

    print(f"  {name}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Generating small/medium matrices...")
    for (rows, cols) in SMALL_MEDIUM_SIZES:
        for suffix in TYPES:
            gen_variant(rows, cols, suffix)

    print("Generating large matrices...")
    for (rows, cols) in LARGE_SIZES:
        gen_variant(rows, cols, "rand_f32")

    # Touch sentinel so build.c knows data is up to date
    sentinel = os.path.join(DATA_ROOT, ".generated")
    open(sentinel, "w").close()
    print(f"Done — wrote sentinel {sentinel}")


if __name__ == "__main__":
    main()
