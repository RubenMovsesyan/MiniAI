#ifndef RLOG_H
#define RLOG_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

typedef enum {
    LL_TRACE,
    LL_DEBUG,
    LL_INFO,
    LL_WARN,
    LL_ERROR,
    LL_FATAL,
} LogLevel;

#define __LOG_WHITE "\033[97m"
#define __LOG_BLUE "\033[94m"
#define __LOG_GREEN "\033[92m"
#define __LOG_YELLOW "\033[93m"
#define __LOG_RED "\033[91m"
#define __LOG_PINK "\033[35m"
#define __LOG_RESET "\033[0m"

#define __FATAL_EXIT 404

static LogLevel __global_log_level = LL_INFO;
static bool __log_verbose = false;

void initLog();
void __Log_impl(LogLevel level, const char* file, uint32_t line, const char* fmt, ...);

#ifdef RLOG_IMPLEMENTATION

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void initLog() {
    char* log_verbose = getenv("LOG_VERBOSE");
    if (log_verbose != nullptr) {
        __log_verbose = true;
    }

    char* log_level = getenv("LOG_LEVEL");
    if (log_level == nullptr) {
        return;
    }

    if (memcmp(log_level, "TRACE", sizeof(char) * 4) == 0) {
        __global_log_level = LL_TRACE;
    } else if (memcmp(log_level, "DEBUG", sizeof(char) * 5) == 0) {
        __global_log_level = LL_DEBUG;
    } else if (memcmp(log_level, "INFO", sizeof(char) * 4) == 0) {
        __global_log_level = LL_INFO;
    } else if (memcmp(log_level, "WARN", sizeof(char) * 4) == 0) {
        __global_log_level = LL_WARN;
    } else if (memcmp(log_level, "ERROR", sizeof(char) * 5) == 0) {
        __global_log_level = LL_ERROR;
    } else if (memcmp(log_level, "FATAL", sizeof(char) * 5) == 0) {
        __global_log_level = LL_FATAL;
    }
}

void __Log_impl(LogLevel level, const char* file, uint32_t line, const char* fmt, ...) {
    if (level < __global_log_level) {
        return;
    }

    if (__log_verbose) {
        switch (level) {
            case LL_TRACE:
                fprintf(stderr, __LOG_WHITE "[TRACE | %s | %d ]: " __LOG_RESET, file, line);
                break;
            case LL_DEBUG:
                fprintf(stderr, __LOG_BLUE "[DEBUG | %s | %d ]: " __LOG_RESET, file, line);
                break;
            case LL_INFO:
                fprintf(stderr, __LOG_GREEN "[INFO | %s | %d ]: " __LOG_RESET, file, line);
                break;
            case LL_WARN:
                fprintf(stderr, __LOG_YELLOW "[WARN | %s | %d ]: " __LOG_RESET, file, line);
                break;
            case LL_ERROR:
                fprintf(stderr, __LOG_RED "[ERROR | %s | %d ]: " __LOG_RESET, file, line);
                break;
            case LL_FATAL:
                fprintf(stderr, __LOG_PINK "[FATAL | %s | %d ]: " __LOG_RESET, file, line);
                break;
            default:
                return;
        }
    } else {
        switch (level) {
            case LL_TRACE:
                fprintf(stderr, __LOG_WHITE "[TRACE]: " __LOG_RESET);
                break;
            case LL_DEBUG:
                fprintf(stderr, __LOG_BLUE "[DEBUG]: " __LOG_RESET);
                break;
            case LL_INFO:
                fprintf(stderr, __LOG_GREEN "[INFO]: " __LOG_RESET);
                break;
            case LL_WARN:
                fprintf(stderr, __LOG_YELLOW "[WARN]: " __LOG_RESET);
                break;
            case LL_ERROR:
                fprintf(stderr, __LOG_RED "[ERROR]: " __LOG_RESET);
                break;
            case LL_FATAL:
                fprintf(stderr, __LOG_PINK "[FATAL]: " __LOG_RESET);
                break;
            default:
                return;
        }
    }

    va_list args;
    va_start(args, fmt);

    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");

    va_end(args);

    if (level == LL_FATAL) {
        exit(__FATAL_EXIT);
    }
}
#endif // RLOG_IMPLEMENTATION

#define RLOG(level, fmt, ...) __Log_impl(level, __FILE__, __LINE__, fmt, ##__VA_ARGS__)

#ifdef __cplusplus
}
#endif

#endif // RLOG_H
