#define RLOG_IMPLEMENTATION
#include <rlog.h>

#include <matrix/matrix.hpp>

#include <common.h>
#include <type_traits>

// ─── CSV utilities ────────────────────────────────────────────────────────────
// Load a CSV file into a host float array. Caller owns the returned pointer.
// Returns nullptr on failure.
f32* csvLoad(const char* path, i32* out_rows, i32* out_cols) {
    // TODO: implement when matrix.cu is ready
    (void)path;
    *out_rows = 0;
    *out_cols = 0;
    return nullptr;
}

// Compare a GPU Matrix against a CSV file of expected values.
// Returns true if all elements are within epsilon.
bool matCheckCSV(const Matrix& result, const char* expected_path, f32 epsilon = 1e-5f) {
    // TODO: implement (cudaMemcpy to host, then compare against csvLoad)
    (void)result;
    (void)expected_path;
    (void)epsilon;
    return true;
}

// ─── Test harness ─────────────────────────────────────────────────────────────
struct Test {
    const char* name;
    void (*fn)();
};

static i32 s_pass = 0;
static i32 s_fail = 0;

static void runTests(Test* tests, i32 count) {
    for (i32 i = 0; i < count; i++) {
        RLOG(LL_INFO, "[ RUN  ] %s", tests[i].name);
        tests[i].fn();
        RLOG(LL_INFO, "[ PASS ] %s", tests[i].name);
        s_pass++;
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

// Verify that expression types compose and chain without compilation errors.
// No GPU operations are called yet — this is a pure compile-time shape check.
void test_expression_types_compile() {
    using MulExpr    = MatrixMulExpr<Matrix, Matrix>;
    using ChainedMul = MatrixMulExpr<MulExpr, Matrix>;
    using AddExpr    = MatrixAddExpr<Matrix, Matrix>;
    using SubExpr    = MatrixSubExpr<Matrix, Matrix>;
    using HadExpr    = MatrixHadamardExpr<Matrix, Matrix>;
    using TrExpr     = MatrixTransposeExpr<Matrix>;
    using ScaleExpr  = MatrixScalarMulExpr<Matrix>;
    using ColAddExpr = MatrixColAddExpr<Matrix, Matrix>;
    using RowAddExpr = MatrixRowAddExpr<Matrix, Matrix>;

    // All expression types must satisfy the same contract
    static_assert(std::is_same_v<decltype(std::declval<Matrix>() * std::declval<Matrix>()), MulExpr>);
    static_assert(std::is_same_v<decltype(std::declval<MulExpr>() * std::declval<Matrix>()), ChainedMul>);
    static_assert(std::is_same_v<decltype(std::declval<Matrix>() + std::declval<Matrix>()), AddExpr>);
    static_assert(std::is_same_v<decltype(std::declval<Matrix>() - std::declval<Matrix>()), SubExpr>);
    static_assert(std::is_same_v<decltype(std::declval<Matrix>().hadamard(std::declval<Matrix>())), HadExpr>);
    static_assert(std::is_same_v<decltype(std::declval<Matrix>().transpose()), TrExpr>);
    static_assert(std::is_same_v<decltype(std::declval<Matrix>() * 2.0f), ScaleExpr>);
    static_assert(std::is_same_v<decltype(std::declval<Matrix>().colAdd(std::declval<Matrix>())), ColAddExpr>);
    static_assert(std::is_same_v<decltype(std::declval<Matrix>().rowAdd(std::declval<Matrix>())), RowAddExpr>);
}

// ─── Main ─────────────────────────────────────────────────────────────────────
int main() {
    initLog();
    Test tests[] = {
        {"expression_types_compile", test_expression_types_compile},
    };

    runTests(tests, sizeof(tests) / sizeof(tests[0]));

    RLOG(LL_INFO, "%d passed, %d failed", s_pass, s_fail);
    return s_fail > 0 ? 1 : 0;
}
