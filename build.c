#include "build.h"
#include <sys/stat.h>

const char* BUILD_DIR = ".build";

// Returns true if path_a is newer than path_b, or if path_b does not exist.
static bool fileIsNewer(const char* path_a, const char* path_b) {
    struct stat sa, sb;
    if (stat(path_a, &sa) != 0) return false;
    if (stat(path_b, &sb) != 0) return true;
    return sa.st_mtime > sb.st_mtime;
}

int main(int argc, char** argv) {
    initLog();
    initArena(4096 * 4096);

    __LinkedObjVec gpu_objects = {0};

    Build build = newBuildWithCompiler(BUILD_DIR, "clang", argc, argv);

    // ── Step 0: Generate CSV test data (skipped if sentinel is up to date) ────
    const char* gen_script  = "src/matrix/tests/gen_test_data.py";
    const char* gen_sentinel = "src/matrix/tests/data/.generated";
    if (fileIsNewer(gen_script, gen_sentinel)) {
        RLOG(LL_INFO, "Generating matrix CSV test data...");
        Command gen_data = newCommand("python3", (char*)gen_script);
        u32 gen_result = cmdExec(&gen_data);
        if (gen_result != 0) {
            RLOG(LL_ERROR, "Matrix CSV data generation failed (exit %u)", gen_result);
        }
    }
    // Generate agg test data
    gen_script  = "src/agg/tests/gen_test_data.py";
    gen_sentinel = "src/agg/tests/data/.generated";
    if (fileIsNewer(gen_script, gen_sentinel)) {
        RLOG(LL_INFO, "Generating agg CSV test data...");
        Command gen_data = newCommand("python3", (char*)gen_script);
        u32 gen_result = cmdExec(&gen_data);
        if (gen_result != 0) {
            RLOG(LL_ERROR, "Agg CSV data generation failed (exit %u)", gen_result);
        }
    }
    // Generate nn test data
    gen_script  = "src/nn/tests/gen_test_data.py";
    gen_sentinel = "src/nn/tests/data/.generated";
    if (fileIsNewer(gen_script, gen_sentinel)) {
        RLOG(LL_INFO, "Generating nn CSV test data...");
        Command gen_data = newCommand("python3", (char*)gen_script);
        u32 gen_result = cmdExec(&gen_data);
        if (gen_result != 0) {
            RLOG(LL_ERROR, "NN CSV data generation failed (exit %u)", gen_result);
        }
    }
    // ─────────────────────────────────────────────────────────────────────────

    // ── Step 1: Compile CUDA matrix implementation ────────────────────────────
    buildStepSetCompiler(&build, "nvcc");
    buildAddInclude(&build, newDirectInclude("src"));
    buildAddInclude(&build, newDirectInclude("devtools"));
    buildAddCompilationFlag(&build, "--gpu-architecture=sm_89");
    buildAddCompilationFlag(&build, "-std=c++20");
    if (build.builtin.mode == Mode_Debug) {
        buildAddCompilationFlag(&build, "-g");
        buildAddCompilationFlag(&build, "-O0");
    }
    VectorPushBack(LinkedObject, &gpu_objects, buildAddObject(&build, newObject("src/matrix/matrix.cu")));
    buildStepSkipLinking(&build);
    buildStep(&build);
    // ─────────────────────────────────────────────────────────────────────────

    // ── Step 1b: Compile agg/ library (nn depends on it — compile before nn) ──
    buildStepSetCompiler(&build, "nvcc");
    buildAddInclude(&build, newDirectInclude("src"));
    buildAddInclude(&build, newDirectInclude("devtools"));
    buildAddCompilationFlag(&build, "--gpu-architecture=sm_89");
    buildAddCompilationFlag(&build, "-std=c++20");
    if (build.builtin.mode == Mode_Debug) {
        buildAddCompilationFlag(&build, "-g");
        buildAddCompilationFlag(&build, "-O0");
    }
    VectorPushBack(LinkedObject, &gpu_objects, buildAddObject(&build, newObject("src/agg/aggregations.cu")));
    buildStepSkipLinking(&build);
    buildStep(&build);
    // ─────────────────────────────────────────────────────────────────────────

    // ── Step 2: Matrix module tests ───────────────────────────────────────────
    buildStepSetCompiler(&build, "nvcc");
    buildAddInclude(&build, newDirectInclude("src"));
    buildAddInclude(&build, newDirectInclude("devtools"));
    buildAddInclude(&build, newDirectInclude("harness"));
    buildAddCompilationFlag(&build, "--gpu-architecture=sm_89");
    buildAddCompilationFlag(&build, "-std=c++20");
    if (build.builtin.mode == Mode_Debug) {
        buildAddCompilationFlag(&build, "-g");
        buildAddCompilationFlag(&build, "-O0");
        buildAddLinkingFlag(&build, "-g");
    }
    buildAddObject(&build, newObject("src/matrix/tests/test_matrix.cu"));
    buildStepSetOutput(&build, "matrix_tests");
    buildAddLinkingFlag(&build, "-L/opt/cuda/lib64");
    buildAddLink(&build, newDirectLink("cudart"));
    buildAddLink(&build, newDirectLink("stdc++"));
    for (usize i = 0; i < gpu_objects.len; i++) {
        buildAddLinkedObject(&build, gpu_objects.items[i]);
    }
    buildStep(&build);
    // ─────────────────────────────────────────────────────────────────────────

    // ── Step 3: Matrix module benchmarks ──────────────────────────────────────
    buildStepSetCompiler(&build, "nvcc");
    buildAddInclude(&build, newDirectInclude("src"));
    buildAddInclude(&build, newDirectInclude("devtools"));
    buildAddInclude(&build, newDirectInclude("harness"));
    buildAddCompilationFlag(&build, "--gpu-architecture=sm_89");
    buildAddCompilationFlag(&build, "-std=c++20");
    if (build.builtin.mode == Mode_Debug) {
        buildAddCompilationFlag(&build, "-g");
        buildAddCompilationFlag(&build, "-O0");
        buildAddLinkingFlag(&build, "-g");
    }
    buildAddObject(&build, newObject("src/matrix/benchmarks/bench_matrix.cu"));
    buildStepSetOutput(&build, "matrix_bench");
    buildAddLinkingFlag(&build, "-L/opt/cuda/lib64");
    buildAddLink(&build, newDirectLink("cudart"));
    buildAddLink(&build, newDirectLink("stdc++"));
    for (usize i = 0; i < gpu_objects.len; i++) {
        buildAddLinkedObject(&build, gpu_objects.items[i]);
    }
    buildStep(&build);
    // ─────────────────────────────────────────────────────────────────────────

    // ── Step 4: Compile NN module implementations ─────────────────────────────
    buildStepSetCompiler(&build, "nvcc");
    buildAddInclude(&build, newDirectInclude("src"));
    buildAddInclude(&build, newDirectInclude("devtools"));
    buildAddCompilationFlag(&build, "--gpu-architecture=sm_89");
    buildAddCompilationFlag(&build, "-std=c++20");
    if (build.builtin.mode == Mode_Debug) {
        buildAddCompilationFlag(&build, "-g");
        buildAddCompilationFlag(&build, "-O0");
    }
    VectorPushBack(LinkedObject, &gpu_objects, buildAddObject(&build, newObject("src/nn/activations.cu")));
    VectorPushBack(LinkedObject, &gpu_objects, buildAddObject(&build, newObject("src/nn/activation_gradients.cu")));
    VectorPushBack(LinkedObject, &gpu_objects, buildAddObject(&build, newObject("src/nn/losses.cu")));
    VectorPushBack(LinkedObject, &gpu_objects, buildAddObject(&build, newObject("src/nn/loss_gradients.cu")));
    buildStepSkipLinking(&build);
    buildStep(&build);
    // ─────────────────────────────────────────────────────────────────────────

    // ── Step 5: NN module tests ───────────────────────────────────────────────
    buildStepSetCompiler(&build, "nvcc");
    buildAddInclude(&build, newDirectInclude("src"));
    buildAddInclude(&build, newDirectInclude("devtools"));
    buildAddInclude(&build, newDirectInclude("harness"));
    buildAddCompilationFlag(&build, "--gpu-architecture=sm_89");
    buildAddCompilationFlag(&build, "-std=c++20");
    if (build.builtin.mode == Mode_Debug) {
        buildAddCompilationFlag(&build, "-g");
        buildAddCompilationFlag(&build, "-O0");
        buildAddLinkingFlag(&build, "-g");
    }
    buildAddObject(&build, newObject("src/nn/tests/test_activations.cu"));
    buildStepSetOutput(&build, "nn_tests");
    buildAddLinkingFlag(&build, "-L/opt/cuda/lib64");
    buildAddLink(&build, newDirectLink("cudart"));
    buildAddLink(&build, newDirectLink("stdc++"));
    for (usize i = 0; i < gpu_objects.len; i++) {
        buildAddLinkedObject(&build, gpu_objects.items[i]);
    }
    buildStep(&build);
    // ─────────────────────────────────────────────────────────────────────────

    // ── Step 6: NN module benchmarks ──────────────────────────────────────────
    buildStepSetCompiler(&build, "nvcc");
    buildAddInclude(&build, newDirectInclude("src"));
    buildAddInclude(&build, newDirectInclude("devtools"));
    buildAddInclude(&build, newDirectInclude("harness"));
    buildAddCompilationFlag(&build, "--gpu-architecture=sm_89");
    buildAddCompilationFlag(&build, "-std=c++20");
    if (build.builtin.mode == Mode_Debug) {
        buildAddCompilationFlag(&build, "-g");
        buildAddCompilationFlag(&build, "-O0");
        buildAddLinkingFlag(&build, "-g");
    }
    buildAddObject(&build, newObject("src/nn/benchmarks/bench_activations.cu"));
    buildStepSetOutput(&build, "nn_bench");
    buildAddLinkingFlag(&build, "-L/opt/cuda/lib64");
    buildAddLink(&build, newDirectLink("cudart"));
    buildAddLink(&build, newDirectLink("stdc++"));
    for (usize i = 0; i < gpu_objects.len; i++) {
        buildAddLinkedObject(&build, gpu_objects.items[i]);
    }
    buildStep(&build);
    // ─────────────────────────────────────────────────────────────────────────

    // ── Step 8: agg/ module tests ────────────────────────────────────────────
    buildStepSetCompiler(&build, "nvcc");
    buildAddInclude(&build, newDirectInclude("src"));
    buildAddInclude(&build, newDirectInclude("devtools"));
    buildAddInclude(&build, newDirectInclude("harness"));
    buildAddCompilationFlag(&build, "--gpu-architecture=sm_89");
    buildAddCompilationFlag(&build, "-std=c++20");
    if (build.builtin.mode == Mode_Debug) {
        buildAddCompilationFlag(&build, "-g");
        buildAddCompilationFlag(&build, "-O0");
        buildAddLinkingFlag(&build, "-g");
    }
    buildAddObject(&build, newObject("src/agg/tests/test_aggregations.cu"));
    buildStepSetOutput(&build, "agg_tests");
    buildAddLinkingFlag(&build, "-L/opt/cuda/lib64");
    buildAddLink(&build, newDirectLink("cudart"));
    buildAddLink(&build, newDirectLink("stdc++"));
    for (usize i = 0; i < gpu_objects.len; i++) {
        buildAddLinkedObject(&build, gpu_objects.items[i]);
    }
    buildStep(&build);
    // ─────────────────────────────────────────────────────────────────────────

    // ── Step 9: agg/ module benchmarks ───────────────────────────────────────
    buildStepSetCompiler(&build, "nvcc");
    buildAddInclude(&build, newDirectInclude("src"));
    buildAddInclude(&build, newDirectInclude("devtools"));
    buildAddInclude(&build, newDirectInclude("harness"));
    buildAddCompilationFlag(&build, "--gpu-architecture=sm_89");
    buildAddCompilationFlag(&build, "-std=c++20");
    if (build.builtin.mode == Mode_Debug) {
        buildAddCompilationFlag(&build, "-g");
        buildAddCompilationFlag(&build, "-O0");
        buildAddLinkingFlag(&build, "-g");
    }
    buildAddObject(&build, newObject("src/agg/benchmarks/bench_aggregations.cu"));
    buildStepSetOutput(&build, "agg_bench");
    buildAddLinkingFlag(&build, "-L/opt/cuda/lib64");
    buildAddLink(&build, newDirectLink("cudart"));
    buildAddLink(&build, newDirectLink("stdc++"));
    for (usize i = 0; i < gpu_objects.len; i++) {
        buildAddLinkedObject(&build, gpu_objects.items[i]);
    }
    buildStep(&build);
    // ─────────────────────────────────────────────────────────────────────────

    // ── Step 10: Compile fused/ library implementations ───────────────────────
    buildStepSetCompiler(&build, "nvcc");
    buildAddInclude(&build, newDirectInclude("src"));
    buildAddInclude(&build, newDirectInclude("devtools"));
    buildAddCompilationFlag(&build, "--gpu-architecture=sm_89");
    buildAddCompilationFlag(&build, "-std=c++20");
    if (build.builtin.mode == Mode_Debug) {
        buildAddCompilationFlag(&build, "-g");
        buildAddCompilationFlag(&build, "-O0");
    }
    VectorPushBack(LinkedObject, &gpu_objects, buildAddObject(&build, newObject("src/fused/softmax_cross_entropy.cu")));
    buildStepSkipLinking(&build);
    buildStep(&build);
    // ─────────────────────────────────────────────────────────────────────────

    // ── Step 11: fused/ module tests ─────────────────────────────────────────
    buildStepSetCompiler(&build, "nvcc");
    buildAddInclude(&build, newDirectInclude("src"));
    buildAddInclude(&build, newDirectInclude("devtools"));
    buildAddInclude(&build, newDirectInclude("harness"));
    buildAddCompilationFlag(&build, "--gpu-architecture=sm_89");
    buildAddCompilationFlag(&build, "-std=c++20");
    if (build.builtin.mode == Mode_Debug) {
        buildAddCompilationFlag(&build, "-g");
        buildAddCompilationFlag(&build, "-O0");
        buildAddLinkingFlag(&build, "-g");
    }
    buildAddObject(&build, newObject("src/fused/tests/test_fused.cu"));
    buildStepSetOutput(&build, "fused_tests");
    buildAddLinkingFlag(&build, "-L/opt/cuda/lib64");
    buildAddLink(&build, newDirectLink("cudart"));
    buildAddLink(&build, newDirectLink("stdc++"));
    for (usize i = 0; i < gpu_objects.len; i++) {
        buildAddLinkedObject(&build, gpu_objects.items[i]);
    }
    buildStep(&build);
    // ─────────────────────────────────────────────────────────────────────────

    // ── Step 12: fused/ module benchmarks ────────────────────────────────────
    buildStepSetCompiler(&build, "nvcc");
    buildAddInclude(&build, newDirectInclude("src"));
    buildAddInclude(&build, newDirectInclude("devtools"));
    buildAddInclude(&build, newDirectInclude("harness"));
    buildAddCompilationFlag(&build, "--gpu-architecture=sm_89");
    buildAddCompilationFlag(&build, "-std=c++20");
    if (build.builtin.mode == Mode_Debug) {
        buildAddCompilationFlag(&build, "-g");
        buildAddCompilationFlag(&build, "-O0");
        buildAddLinkingFlag(&build, "-g");
    }
    buildAddObject(&build, newObject("src/fused/benchmarks/bench_fused.cu"));
    buildStepSetOutput(&build, "fused_bench");
    buildAddLinkingFlag(&build, "-L/opt/cuda/lib64");
    buildAddLink(&build, newDirectLink("cudart"));
    buildAddLink(&build, newDirectLink("stdc++"));
    for (usize i = 0; i < gpu_objects.len; i++) {
        buildAddLinkedObject(&build, gpu_objects.items[i]);
    }
    buildStep(&build);
    // ─────────────────────────────────────────────────────────────────────────

    // ── Step 13: Compile mlkit/ library implementations ──────────────────────
    buildStepSetCompiler(&build, "nvcc");
    buildAddInclude(&build, newDirectInclude("src"));
    buildAddInclude(&build, newDirectInclude("devtools"));
    buildAddCompilationFlag(&build, "--gpu-architecture=sm_89");
    buildAddCompilationFlag(&build, "-std=c++20");
    if (build.builtin.mode == Mode_Debug) {
        buildAddCompilationFlag(&build, "-g");
        buildAddCompilationFlag(&build, "-O0");
    }
    VectorPushBack(LinkedObject, &gpu_objects, buildAddObject(&build, newObject("src/mlkit/init.cu")));
    buildStepSkipLinking(&build);
    buildStep(&build);
    // ─────────────────────────────────────────────────────────────────────────

    // ── Step 14: mlkit/ module tests (no benchmark — host-side startup code) ──
    buildStepSetCompiler(&build, "nvcc");
    buildAddInclude(&build, newDirectInclude("src"));
    buildAddInclude(&build, newDirectInclude("devtools"));
    buildAddInclude(&build, newDirectInclude("harness"));
    buildAddCompilationFlag(&build, "--gpu-architecture=sm_89");
    buildAddCompilationFlag(&build, "-std=c++20");
    if (build.builtin.mode == Mode_Debug) {
        buildAddCompilationFlag(&build, "-g");
        buildAddCompilationFlag(&build, "-O0");
        buildAddLinkingFlag(&build, "-g");
    }
    buildAddObject(&build, newObject("src/mlkit/tests/test_mlkit.cu"));
    buildStepSetOutput(&build, "mlkit_tests");
    buildAddLinkingFlag(&build, "-L/opt/cuda/lib64");
    buildAddLink(&build, newDirectLink("cudart"));
    buildAddLink(&build, newDirectLink("stdc++"));
    for (usize i = 0; i < gpu_objects.len; i++) {
        buildAddLinkedObject(&build, gpu_objects.items[i]);
    }
    // ─────────────────────────────────────────────────────────────────────────

    buildBuild(&build);
    freeArena();
    return 0;
}
