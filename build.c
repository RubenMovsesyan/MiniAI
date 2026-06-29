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
        RLOG(LL_INFO, "Generating CSV test data...");
        Command gen_data = newCommand("python3", (char*)gen_script);
        u32 gen_result = cmdExec(&gen_data);
        if (gen_result != 0) {
            RLOG(LL_ERROR, "CSV data generation failed (exit %u)", gen_result);
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
