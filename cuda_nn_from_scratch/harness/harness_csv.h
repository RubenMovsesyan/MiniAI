#pragma once

#include <common.h>
#include <rlog.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

// Load a CSV of floats into a host array. Dimensions inferred: rows = line count,
// cols = comma-delimited tokens on the first line. Caller owns the returned buffer (free()).
inline f32* csvLoad(const char* path, i32* out_rows, i32* out_cols) {
    FILE* f = fopen(path, "r");
    if (!f) { RLOG(LL_ERROR, "csvLoad: cannot open %s", path); return nullptr; }

    char* line = (char*)malloc(1 << 20);
    i32 rows = 0, cols = 0;
    while (fgets(line, 1 << 20, f)) {
        if (rows == 0) {
            cols = 1;
            for (char* p = line; *p; p++) if (*p == ',') cols++;
        }
        rows++;
    }
    rewind(f);

    f32* data = (f32*)malloc((usize)rows * cols * sizeof(f32));
    i32 idx = 0;
    while (fgets(line, 1 << 20, f)) {
        char* tok = strtok(line, ",\n\r");
        while (tok) { data[idx++] = (f32)atof(tok); tok = strtok(nullptr, ",\n\r"); }
    }
    fclose(f);
    free(line);
    *out_rows = rows;
    *out_cols = cols;
    return data;
}
