#pragma once

#include <common.h>
#include <rlog.h>

// Minimal pass/fail counters shared across translation units (C++20 inline vars).

inline i32 s_pass = 0;
inline i32 s_fail = 0;

inline void record(bool ok, const char* label) {
    if (ok) { RLOG(LL_INFO,  "[ PASS ] %s", label); s_pass++; }
    else    { RLOG(LL_ERROR, "[ FAIL ] %s", label); s_fail++; }
}

// Print the final tally; return a process exit code (non-zero if any test failed).
inline i32 testSummary() {
    RLOG(LL_INFO, "%d passed, %d failed", s_pass, s_fail);
    return s_fail > 0 ? 1 : 0;
}
