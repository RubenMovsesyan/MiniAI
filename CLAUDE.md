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
    <module_name>.hpp     ← public interface
    <module_name>.cu      ← CUDA implementation (when applicable)
    tests/
      test_<module_name>.cpp   ← test runner
      data/                    ← CSV fixtures for input/output tests
```

- One test executable per module, compiled to `.build/<module>_tests`
- Add each new module's test step to `build.c` following the matrix tests pattern
- Module schema is not final — update this file as new modules are added

Current modules:
- `matrix/` — GPU matrix math (CUDA)

## Testing Conventions

Tests use a minimal hand-rolled harness (no external test framework). Each module has a dedicated test binary.

**CSV-driven tests** (used for matrix ops): test functions load CSV files from `tests/data/` as matrix inputs and expected outputs, run the GPU operation, and compare results within a float tolerance. Utilities: `csvLoad()` (host CSV → float array) and `matCheckCSV()` (GPU Matrix vs CSV). These are defined in `src/matrix/tests/test_matrix.cpp` and should be replicated/generalized when other modules need them.

Run matrix tests:
```bash
./build && .build/matrix_tests
```

## Matrix Design

`src/matrix/matrix.hpp` uses **expression templates** for lazy GPU evaluation:
- `Matrix` is the concrete type (holds a GPU device pointer)
- Operations (`*`, `+`, `-`, etc.) return lightweight expression types that hold references to their operands — no GPU work happens yet
- GPU evaluation is triggered by `Matrix::operator=(const Expr&)` or `.eval()`
- Example: `C = A * B * D.transpose()` — the full chain evaluates in one assign

Expression type lifetime: expressions hold const-refs, so they must be evaluated within the same full-expression. Storing in `auto` and using later is UB for chained temporaries.

## Language & Conventions

- **C++20** throughout: source files, test files, and nvcc (max supported by nvcc)
- **C** for `build.c` only
- **GPU target**: sm_89 (RTX 40-series / Ada Lovelace), CUDA at `/opt/cuda`
- **Type aliases**: use `u8`, `u16`, `u32`, `u64`, `usize`, `i8`–`i64`, `isize`, `f32`, `f64` from `<common.h>`
- **Includes**: angle-bracket style with module paths — `#include <matrix/matrix.hpp>`, `#include <common.h>`
- **Logging**: `RLOG(LL_INFO, "...")` from `<rlog.h>`; define `RLOG_IMPLEMENTATION` in exactly one TU per binary
