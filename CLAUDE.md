# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

MiniAI is a repository for experimenting with AI, neural networks, and the math behind them. The goal is to eventually train AI models from scratch, starting with GPU math primitives built on CUDA.

## Build System

The project uses the same custom C build system as `~/dev/AstroPhoto` (build.h, not CMake/Make).

```bash
# Bootstrap the build tool (only needed once, or when build.c/build.h change)
clang -std=c23 -o build build.c

# Build (debug)
./build

# Build (release)
./build -Release
```

`build.c` is the build configuration. `build.h` is the build system library — do not edit it.
Build output goes to `.build/`. The build system auto-generates `compile_commands.json` for LSP support.

## Module Schema

Each functional area of the project lives under its own directory in `src/`:

```
src/
  <module_name>/
    <module_name>.cuh     ← public interface (use .hpp for pure C++ modules)
    <module_name>.cu      ← CUDA implementation (when applicable)
    tests/
      test_<module_name>.cu    ← test runner
      gen_test_data.py         ← generates CSV fixtures (run via ./build)
      data/                    ← CSV fixtures (gitignored, generated)
    benchmarks/
      bench_<module_name>.cu   ← benchmark runner
      baseline.csv             ← timing baselines (gitignored, auto-created)
```

Shared test/benchmark code lives in the root-level `harness/` directory (included directly,
like `devtools/`, via `-Iharness`; headers are `harness_`-prefixed, included as `<harness_*.h>`):
- `harness_csv.h` — `csvLoad()` (CSV → host float array), generic
- `harness_test.h` — `record()` / `testSummary()` pass-fail counters, generic
- `harness_bench.cuh` — generic benchmark framework (timing, stats, baseline regression)
- `harness_matrix_csv.cuh` — matrix glue: `matLoad()`, `matCheckCSV()`, variant lists, path macros

Generic harness headers have **no** matrix dependency; `harness_matrix_csv.cuh` depends on the
matrix lib one-way (`matrix.cuh` never includes harness, so no cycle).

- One test executable per module, compiled to `.build/<module>_tests`; one benchmark executable
  compiled to `.build/<module>_bench`
- Add each new module's test + benchmark steps to `build.c` following the matrix pattern. Finalize
  each non-final build step with `buildStep(&build)`; `buildBuild()` finalizes the last one
- Module schema is not final — update this file as new modules are added

Current modules:
- `matrix/` — GPU matrix math (CUDA)
- `nn/` — Neural network primitives: activations (relu, softmax, etc.) + gradients + losses (cross_entropy) (CUDA)
- `agg/` — Aggregation operations: sum/max reductions over rows/columns (CUDA)
- `fused/` — Fused forward+backward math shortcuts that couple primitives (CUDA)
- `mlkit/` — Miscellaneous ML utilities (host-side); first feature: weight initialization

## Testing Conventions

Tests use a minimal hand-rolled harness (no external test framework). Each module has a dedicated test binary.

**CSV-driven tests** (used for matrix ops): test functions load CSV files from `tests/data/` as matrix inputs and expected outputs, run the GPU operation, and compare results within a float tolerance. Utilities: `csvLoad()` (host CSV → float array, in `harness_csv.h`) and `matLoad()`/`matCheckCSV()` (GPU Matrix vs CSV, in `harness_matrix_csv.cuh`). `matCheckCSV` uses a **relative** tolerance (`epsilon * max(1, |expected|)`) so large structured matrices pass without a huge absolute epsilon.

**Test data layout** (generated, not committed — see `.gitignore`):
```
src/matrix/tests/data/
  inputs/{rows}x{cols}_{type}/   A.csv, B.csv, col.csv, row.csv
  expected/{op}/{rows}x{cols}_{type}.csv
```
`{op}` is `add`, `sub`, `hadamard`, `transpose`, `scalar_x{val}`, `colAdd`, `rowAdd` — or any chained operation name added later.

**Data generation** is run automatically by `./build` when `gen_test_data.py` is newer than `data/.generated`. To regenerate manually:
```bash
python3 src/matrix/tests/gen_test_data.py
```

Run matrix tests:
```bash
./build && .build/matrix_tests
```

## Benchmarking

Benchmarks reuse the test CSVs but **time** the ops instead of checking correctness. The generic
framework is `harness_bench.cuh`: `benchmark(cfg, func, lists...)` times `func(lists[i]...)` over
rows of per-parameter input lists (one `std::vector` per function arg). Full cycle per row: warmup →
`iterations` timed runs with CUDA events and an L2-cache flush between runs → min/median/mean/max/σ.

- **Modes**: `BenchMode::Sequential` (default — one row fully, then next) or `RoundRobin` (interleave
  rows each iteration; all inputs resident).
- **Regression**: each run compares the row's median against `baseline.csv` (per-consumer, gitignored).
  Missing rows are recorded as the new baseline; existing rows that are `> epsilon` (default 10%)
  slower trigger `RLOG(LL_WARN, "SLOWDOWN ...")`. Baselines are never auto-overwritten — rebaseline
  with `BENCH_REBASELINE=1` or by deleting `baseline.csv`.

Run matrix benchmarks (first run writes the baseline; later runs compare):
```bash
./build && .build/matrix_bench
BENCH_REBASELINE=1 .build/matrix_bench   # reset baselines after an intended perf change
```

## Matrix Design

`src/matrix/matrix.cuh` uses **expression templates** for lazy GPU evaluation:
- `Matrix` is the concrete type (holds a GPU device pointer); implemented in `matrix.cu`
- Operations (`*`, `+`, `-`, etc.) return lightweight expression types that store operands **by value** — safe to pass to CUDA kernels; no GPU work happens at construction
- GPU evaluation is triggered by `Matrix::operator=(const Expr&)`, which launches `matEvalKernel` — a single fused kernel that calls `expr(r, c)` per thread, recursively evaluating the full expression tree at compile time with no intermediate allocations
- Example: `C = A + B * 2.0f` — one kernel launch, zero temporary matrices
- Matrix multiply needs a K-reduction, so it can't fuse element-wise. `MatrixMulExpr` is routed to a dedicated `gemmKernel` (in `matrix.cu`) via a specialized `Matrix::operator=` overload. When a matmul appears as an **inner** node of an element-wise tree, `materialize()`/`materializeToRef()` (in `matrix.cuh`) pre-evaluate each matmul subtree into its own GPU buffer (owned by a temp `std::vector<Matrix>`) and splice in a `MatrixRef`, so the surrounding element-wise ops still fuse into one `matEvalKernel`

**Two ways to use the ops — lazy (fused) and eager (runtime):**
- **Plain-verb methods on `Matrix` are eager** — they run immediately and return an owned `Matrix`, so they chain: `A.matmul(B).matmul(C)`, `A.add(B).scale(2.0f)`. Set: `matmul, add, sub, hadamard, scale, transposed, colAdd, rowAdd`. Each has an out-param overload (`A.matmul(B, out)`) that writes into a preallocated `Matrix` (caller-supplied correct shape) to avoid allocation. Bodies just build the matching lazy node and force it via `operator=`, so each eager op is still one fused kernel internally; only *between* chained steps is there a temporary.
- **Operators (`+ - *`) stay lazy**, and `A.lazy()` (alias of `ref()`) is the gateway into the deferred expression world: `A.lazy().mul(B).add(C).eval()`. The named lazy adapters (`mul/add/sub/scale/hadamard/transpose/colAdd/rowAdd`) live on the expression view; `MatrixExpr::eval()` is the terminal that forces a tree into an owned `Matrix`.
- Rule of thumb: **plain verb = do it now; `.lazy()`/operators = deferred & fused.** `transposed()` (eager, returns new — Rust `sorted`/`reversed` style) vs lazy `transpose()` (on the `.lazy()` view) is the only name that splits.
- Note `MatrixExpr::eval()` (force an expression → new `Matrix`) is distinct from `Matrix::eval()` (clone an existing `Matrix` device→device).

Header extension: `.cuh` for modules with CUDA `__device__` code, `.hpp` for pure C++ modules.

## NN Module (Activations & Losses)

`src/nn/` provides neural network building blocks: activation functions and loss functions, all GPU-based.

**Activations** (both eager and lazy evaluation):
- **Eager path**: `Matrix y = relu(x);` — runs immediately on GPU, returns an owned `Matrix`.
- **Lazy path**: `ReluExpr<MatrixRef> expr = ReluExpr<MatrixRef>(x.ref()); Matrix y = expr.eval();` — deferred, can fuse with other ops in one kernel.
- Implemented: `relu` + `grad_relu` only (others are stubs with RLOG_WARN).
- `softmax` implemented (dedicated per-row kernel, numerically stable via max-subtraction; lazy path materializes like matmul). Framework stubs: `sigmoid`, `bipolar_sigmoid`, `tanh`, `leaky_relu`, `step`, `threshold` (forward + gradients).

**Losses**:
- `cross_entropy(logits, targets)` implemented — `a2 = softmax(logits)`, then `-(1/N) Σ targets·log(a2 + 1e-9)` reduced to `Matrix(1,1)` on device (targets are one-hot; caller syncs + reads back to host when logging). Uses `agg`'s `col_sum`.
- `mse`, `l1_loss`, `l2_loss` (forward) + `grad_mse`, `grad_l1_loss`, `grad_l2_loss` (backward) still stubs.
- The CE gradient is fused with softmax — lives in `fused/`, not here (see Fused Module).

**Expression types** (in `activations.cuh`):
- `ReluExpr<LHS>`, `SigmoidExpr<LHS>`, etc. — each stores its input and implements `__device__ operator()` for element-wise computation.
- Inherit from `MatrixExpr` so they compose with the matrix expression tree.
- Lazy path: manually build tree and call `.eval()` (eager methods on `Matrix` forward to C++ functions; lazy methods return expression types).

**Data stays on GPU**: no host↔device copies within eager or lazy evaluation.

## Aggregation Module (agg/)

`src/agg/` provides GPU-based reduction operations (sum aggregations).

**Operations** (both eager and lazy evaluation):
- **Eager path**: `Matrix y = row_sum(x);` — runs immediately on GPU, returns owned `Matrix`
- **Lazy path**: `RowSumExpr<MatrixRef> expr = RowSumExpr<MatrixRef>(x.ref()); Matrix y = expr.eval();` — deferred, can fuse with other ops
- Implemented: `row_sum`, `col_sum`, `sum` (chains reductions)
- Out-param overloads: `row_sum(x, out)` writes into preallocated buffer

**Reduction kernels**:
- Use block-level shared-memory reduction via `__syncthreads()` (not atomicAdd)
- One block per row (row_sum) or per column (col_sum); each block cooperatively reduces via striding + shared mem
- `sum(x)` = `col_sum(row_sum(x))` — two sequential kernels in eager path

**Expression types** (in `aggregations.cuh`):
- `RowSumExpr<LHS>`, `ColSumExpr<LHS>` — store input, implement `__device__ operator()(r,c)` with reduction loop
- Inherit from `MatrixExpr` so they compose with matrix expression trees

**Output shape**:
- `row_sum(Matrix(rows, cols))` → `Matrix(rows, 1)`
- `col_sum(Matrix(rows, cols))` → `Matrix(1, cols)`
- `sum(Matrix(rows, cols))` → `Matrix(1, 1)`
- Allows chaining: `row_sum(x).col_sum()` valid

**Data stays on GPU**: no host↔device copies within operations.

## Fused Module (fused/)

`src/fused/` holds ops that couple a forward+backward math shortcut across primitives —
cases where composing two primitives collapses to something far cheaper than the naive
chain. Depends on `nn` and `matrix`.

- **`grad_softmax_cross_entropy(logits, targets)`** — the CE-w.r.t.-logits gradient. The
  softmax Jacobian cancels against the cross-entropy derivative, so the whole backward pass
  is `(softmax(logits) − targets) / N` (one-hot `targets`, `N = rows`). Pure element-wise:
  runs `softmax`, then one fused `matEvalKernel` (sub + scalar-mul) via the expression engine.
  Eager owned-return + out-param overloads.
- **Correctness gate**: `fused_tests` gradient-checks the analytic gradient against central
  finite differences of `cross_entropy` — the most important check in the project.

## mlkit Module (mlkit/)

`src/mlkit/` holds miscellaneous ML utilities that aren't core GPU primitives (weight
init now; metrics, schedulers, data shuffling, etc. later). Depends only on `matrix`.

**Weight initialization** (`init.cuh`) — variance-scaled init keeps activation variance
~constant through depth (flat ±0.5 init explodes gradients). In-place fill on a
preallocated `Matrix`; host-side `<random>` generation + one host→device `cudaMemcpy`
(no kernel — it runs once at startup, so **no benchmark**, tests only).

- Schemes (each with `_normal` and `_uniform` variant): **He** (ReLU, `var=2/fan_in`),
  **LeCun** (linear/SELU, `var=1/fan_in`), **Xavier/Glorot** (tanh/sigmoid,
  `var=2/(fan_in+fan_out)`). Uniform limits: `√(6/·)`, `√(3/fan_in)`, `√(6/·)`.
- Each has an explicit-param form (`he_normal(w, fan_in)`, `xavier_normal(w, fan_in, fan_out)`)
  and a shape-deriving overload assuming layout `W = fan_in × fan_out` (`fan_in=rows`, `fan_out=cols`).
- `fill_normal`/`fill_uniform` primitives, `zero_init(w)` for biases.
- `mlkit_seed(u32)` seeds the module-level `mt19937` for reproducible weights (tests rely on it).

## Language & Conventions

- **C++20** throughout: source files, test files, and nvcc (max supported by nvcc)
- **C** for `build.c` only
- **GPU target**: sm_89 (RTX 40-series / Ada Lovelace), CUDA at `/opt/cuda`
- **Type aliases**: always use `f32`/`f64` instead of `float`/`double`; `i32`/`u32` etc. instead of `int`/`unsigned`; `usize`/`isize` for sizes — all from `<common.h>`
- **No raw C primitives**: `float`, `double`, `int`, `unsigned` are banned in source and header files; use the aliases above
- **Printing**: use `RLOG(LL_INFO/LL_WARN/LL_ERROR, "...")` from `<rlog.h>` — never `printf` or `fprintf`; call `initLog()` once in `main()`; define `RLOG_IMPLEMENTATION` in exactly one TU per binary
- **Includes**: angle-bracket style with module paths — `#include <matrix/matrix.hpp>`, `#include <common.h>`
