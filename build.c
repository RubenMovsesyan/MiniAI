#include "build.h"

const char* BUILD_DIR = ".build";
static char* COMPILER = "clang++";

int main(int argc, char** argv) {
    initLog();
    initArena(4096 * 4096);

    // gpu_objects collects compiled .cu object handles to link into test binaries
    __LinkedObjVec gpu_objects = {0};

    Build build = newBuildWithCompiler(BUILD_DIR, COMPILER, argc, argv);

    // ── CUDA step — uncomment when src/matrix/matrix.cu exists ───────────────
    // buildStepSetCompiler(&build, "nvcc");
    // buildAddInclude(&build, newDirectInclude("src"));
    // buildAddInclude(&build, newDirectInclude("devtools"));
    // buildAddCompilationFlag(&build, "--gpu-architecture=sm_89");
    // buildAddCompilationFlag(&build, "-std=c++20");
    // if (build.builtin.mode == Mode_Debug) {
    //     buildAddCompilationFlag(&build, "-g");
    //     buildAddCompilationFlag(&build, "-O0");
    // }
    // VectorPushBack(LinkedObject, &gpu_objects, buildAddObject(&build, newObject("src/matrix/matrix.cu")));
    // buildStepSkipLinking(&build);
    // buildStep(&build);
    // ─────────────────────────────────────────────────────────────────────────

    // ── Matrix module tests ───────────────────────────────────────────────────
    buildAddInclude(&build, newDirectInclude("src"));
    buildAddInclude(&build, newDirectInclude("devtools"));
    buildAddCompilationFlag(&build, "-std=c++20");
    if (build.builtin.mode == Mode_Debug) {
        buildAddCompilationFlag(&build, "-g");
        buildAddCompilationFlag(&build, "-O0");
        buildAddLinkingFlag(&build, "-g");
    }
    buildAddObject(&build, newObject("src/matrix/tests/test_matrix.cpp"));
    buildStepSetOutput(&build, "matrix_tests");
    buildAddLink(&build, newDirectLink("stdc++"));
    // Link CUDA runtime and GPU objects once matrix.cu is compiled:
    // buildAddLink(&build, newPathLinkWithDep("cudart", "/opt/cuda/lib64"));
    // for (usize i = 0; i < gpu_objects.len; i++) {
    //     buildAddLinkedObject(&build, gpu_objects.items[i]);
    // }
    // ─────────────────────────────────────────────────────────────────────────

    buildBuild(&build);
    freeArena();
    return 0;
}
