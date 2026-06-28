#include "build.h"

const char* BUILD_DIR = ".build";

int main(int argc, char** argv) {
    initLog();
    initArena(4096 * 4096);

    __LinkedObjVec gpu_objects = {0};

    Build build = newBuildWithCompiler(BUILD_DIR, "clang", argc, argv);

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

    // ── Step 2: Matrix module tests ───────────────────────────────────────────
    buildStepSetCompiler(&build, "nvcc");
    buildAddInclude(&build, newDirectInclude("src"));
    buildAddInclude(&build, newDirectInclude("devtools"));
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
    // ─────────────────────────────────────────────────────────────────────────

    buildBuild(&build);
    freeArena();
    return 0;
}
