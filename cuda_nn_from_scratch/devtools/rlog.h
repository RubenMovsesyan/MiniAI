#ifndef RLOG_H
#define RLOG_H

#ifdef __cplusplus
#include <atomic>
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

void initLog(uint32_t buffer_size);
void __Log_impl(LogLevel level, const char* file, uint32_t line, const char* fmt, ...);
void __Log_file_impl(const char* path, LogLevel level, const char* file, uint32_t line, const char* fmt, ...);

#ifdef RLOG_IMPLEMENTATION

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#ifdef __cplusplus
#define RUNNING_LOAD(x) ((x).load())
#define RUNNING_STORE(x, v) ((x).store(v))
#define RUNNING_INIT(x, v) ((x).store(v))
#else
#define RUNNING_LOAD(x) (x)
#define RUNNING_STORE(x, v) ((x) = (v))
#define RUNNING_INIT(x, v) ((x) = (v))
#endif

typedef struct {
    char*           buf;
    uint32_t        capacity;
    uint32_t        write_pos;
    uint32_t        read_pos;
    uint32_t        data_len;
    pthread_mutex_t mutex;
    pthread_cond_t  cond;
    pthread_t       thread;
#ifdef __cplusplus
    std::atomic<bool> running;
#else
    volatile bool running;
#endif
} __RLogState;

#ifndef RLOG_MAX_FILES
#define RLOG_MAX_FILES 8
#endif

typedef struct {
    char  path[256];
    FILE* fp;
} __RLogFileEntry;

static __RLogFileEntry __rlog_files[RLOG_MAX_FILES];
static uint32_t        __rlog_file_count = 0;

static __RLogState __rlog_state;

static void* __rlog_thread_fn(void* arg) {
    (void)arg;
    __RLogState* s = &__rlog_state;

    while (true) {
        pthread_mutex_lock(&s->mutex);

        while (s->data_len == 0 && RUNNING_LOAD(s->running))
            pthread_cond_wait(&s->cond, &s->mutex);

        while (s->data_len > 0) {
            if (s->read_pos + 4 > s->capacity) {
                s->data_len -= s->capacity - s->read_pos;
                s->read_pos = 0;
                continue;
            }

            uint32_t raw_len;
            memcpy(&raw_len, s->buf + s->read_pos, 4);

            if (raw_len == UINT32_MAX) {
                s->data_len -= s->capacity - s->read_pos;
                s->read_pos = 0;
                continue;
            }

            bool is_file = (raw_len >> 31) & 1;
            uint32_t len = raw_len & 0x7FFFFFFFu;
            uint32_t header_size = 4;
            FILE* fp = stderr;

            if (is_file) {
                memcpy(&fp, s->buf + s->read_pos + 4, sizeof(FILE*));
                header_size = 4 + (uint32_t)sizeof(FILE*);
            }

            s->data_len -= header_size + len;
            s->read_pos += header_size;
            fwrite(s->buf + s->read_pos, 1, len, fp);
            s->read_pos += len;
            if (s->read_pos == s->capacity)
                s->read_pos = 0;
        }

        bool still_running = RUNNING_LOAD(s->running);
        pthread_mutex_unlock(&s->mutex);

        if (!still_running)
            break;
    }

    return NULL;
}

static void __rlog_shutdown(void) {
    __RLogState* s = &__rlog_state;
    pthread_mutex_lock(&s->mutex);
    RUNNING_STORE(s->running, false);
    pthread_cond_signal(&s->cond);
    pthread_mutex_unlock(&s->mutex);
    pthread_join(s->thread, NULL);
    free(s->buf);
    pthread_mutex_destroy(&s->mutex);
    pthread_cond_destroy(&s->cond);
    for (uint32_t i = 0; i < __rlog_file_count; i++) {
        if (__rlog_files[i].fp != NULL) {
            fclose(__rlog_files[i].fp);
        }
    }
}

void initLog(uint32_t buffer_size) {
    char* log_verbose = getenv("LOG_VERBOSE");
    if (log_verbose != nullptr) {
        __log_verbose = true;
    }

    char* log_level = getenv("LOG_LEVEL");
    if (log_level != nullptr) {
        if (memcmp(log_level, "TRACE", sizeof(char) * 5) == 0) {
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

    __RLogState* s = &__rlog_state;
    s->buf = (char *)malloc(buffer_size);
    s->capacity = buffer_size;
    s->write_pos = s->read_pos = s->data_len = 0;
    pthread_mutex_init(&s->mutex, NULL);
    pthread_cond_init(&s->cond, NULL);
    RUNNING_INIT(s->running, true);
    pthread_create(&s->thread, NULL, __rlog_thread_fn, NULL);
    atexit(__rlog_shutdown);
}

void __Log_impl(LogLevel level, const char* file, uint32_t line, const char* fmt, ...) {
    if (level < __global_log_level) {
        return;
    }

    char msg[1024];
    int prefix_len;

    if (__log_verbose) {
        switch (level) {
            case LL_TRACE:
                prefix_len = snprintf(msg, sizeof(msg), __LOG_WHITE "[TRACE | %s | %d ]: " __LOG_RESET, file, line);
                break;
            case LL_DEBUG:
                prefix_len = snprintf(msg, sizeof(msg), __LOG_BLUE "[DEBUG | %s | %d ]: " __LOG_RESET, file, line);
                break;
            case LL_INFO:
                prefix_len = snprintf(msg, sizeof(msg), __LOG_GREEN "[INFO | %s | %d ]: " __LOG_RESET, file, line);
                break;
            case LL_WARN:
                prefix_len = snprintf(msg, sizeof(msg), __LOG_YELLOW "[WARN | %s | %d ]: " __LOG_RESET, file, line);
                break;
            case LL_ERROR:
                prefix_len = snprintf(msg, sizeof(msg), __LOG_RED "[ERROR | %s | %d ]: " __LOG_RESET, file, line);
                break;
            case LL_FATAL:
                prefix_len = snprintf(msg, sizeof(msg), __LOG_PINK "[FATAL | %s | %d ]: " __LOG_RESET, file, line);
                break;
            default:
                return;
        }
    } else {
        switch (level) {
            case LL_TRACE:
                prefix_len = snprintf(msg, sizeof(msg), __LOG_WHITE "[TRACE]: " __LOG_RESET);
                break;
            case LL_DEBUG:
                prefix_len = snprintf(msg, sizeof(msg), __LOG_BLUE "[DEBUG]: " __LOG_RESET);
                break;
            case LL_INFO:
                prefix_len = snprintf(msg, sizeof(msg), __LOG_GREEN "[INFO]: " __LOG_RESET);
                break;
            case LL_WARN:
                prefix_len = snprintf(msg, sizeof(msg), __LOG_YELLOW "[WARN]: " __LOG_RESET);
                break;
            case LL_ERROR:
                prefix_len = snprintf(msg, sizeof(msg), __LOG_RED "[ERROR]: " __LOG_RESET);
                break;
            case LL_FATAL:
                prefix_len = snprintf(msg, sizeof(msg), __LOG_PINK "[FATAL]: " __LOG_RESET);
                break;
            default:
                return;
        }
    }

    if (prefix_len < 0 || prefix_len >= (int)sizeof(msg)) {
        prefix_len = 0;
    }

    va_list args;
    va_start(args, fmt);
    int body_len = vsnprintf(msg + prefix_len, sizeof(msg) - (uint32_t)prefix_len, fmt, args);
    va_end(args);

    if (body_len < 0) {
        body_len = 0;
    }

    int total = prefix_len + body_len;
    if (total >= (int)sizeof(msg) - 1) {
        total = (int)sizeof(msg) - 2;
    }
    msg[total] = '\n';
    total += 1;

    uint32_t msg_len = (uint32_t)total;
    uint32_t entry_size = 4 + msg_len;

    __RLogState* s = &__rlog_state;
    pthread_mutex_lock(&s->mutex);

    uint32_t space_at_end = s->capacity - s->write_pos;

    if (space_at_end < entry_size) {
        if (s->data_len + space_at_end + entry_size > s->capacity) {
            pthread_mutex_unlock(&s->mutex);
            goto fatal_check;
        }
        if (space_at_end >= 4) {
            uint32_t sentinel = UINT32_MAX;
            memcpy(s->buf + s->write_pos, &sentinel, 4);
        }
        s->data_len += space_at_end;
        s->write_pos = 0;
    }

    if (s->data_len + entry_size > s->capacity) {
        pthread_mutex_unlock(&s->mutex);
        goto fatal_check;
    }

    memcpy(s->buf + s->write_pos, &msg_len, 4);
    memcpy(s->buf + s->write_pos + 4, msg, msg_len);
    s->write_pos += entry_size;
    if (s->write_pos == s->capacity)
        s->write_pos = 0;
    s->data_len += entry_size;

    pthread_cond_signal(&s->cond);
    pthread_mutex_unlock(&s->mutex);

fatal_check:
    if (level == LL_FATAL) {
        exit(__FATAL_EXIT);
    }
}

void __Log_file_impl(const char* path, LogLevel level, const char* file, uint32_t line, const char* fmt, ...) {
    if (level < __global_log_level) {
        return;
    }

    char msg[1024];
    int prefix_len;

    if (__log_verbose) {
        switch (level) {
            case LL_TRACE:
                prefix_len = snprintf(msg, sizeof(msg), "[TRACE | %s | %d ]: ", file, line);
                break;
            case LL_DEBUG:
                prefix_len = snprintf(msg, sizeof(msg), "[DEBUG | %s | %d ]: ", file, line);
                break;
            case LL_INFO:
                prefix_len = snprintf(msg, sizeof(msg), "[INFO | %s | %d ]: ", file, line);
                break;
            case LL_WARN:
                prefix_len = snprintf(msg, sizeof(msg), "[WARN | %s | %d ]: ", file, line);
                break;
            case LL_ERROR:
                prefix_len = snprintf(msg, sizeof(msg), "[ERROR | %s | %d ]: ", file, line);
                break;
            case LL_FATAL:
                prefix_len = snprintf(msg, sizeof(msg), "[FATAL | %s | %d ]: ", file, line);
                break;
            default:
                return;
        }
    } else {
        switch (level) {
            case LL_TRACE:
                prefix_len = snprintf(msg, sizeof(msg), "[TRACE]: ");
                break;
            case LL_DEBUG:
                prefix_len = snprintf(msg, sizeof(msg), "[DEBUG]: ");
                break;
            case LL_INFO:
                prefix_len = snprintf(msg, sizeof(msg), "[INFO]: ");
                break;
            case LL_WARN:
                prefix_len = snprintf(msg, sizeof(msg), "[WARN]: ");
                break;
            case LL_ERROR:
                prefix_len = snprintf(msg, sizeof(msg), "[ERROR]: ");
                break;
            case LL_FATAL:
                prefix_len = snprintf(msg, sizeof(msg), "[FATAL]: ");
                break;
            default:
                return;
        }
    }

    if (prefix_len < 0 || prefix_len >= (int)sizeof(msg)) {
        prefix_len = 0;
    }

    va_list args;
    va_start(args, fmt);
    int body_len = vsnprintf(msg + prefix_len, sizeof(msg) - (uint32_t)prefix_len, fmt, args);
    va_end(args);

    if (body_len < 0) {
        body_len = 0;
    }

    int total = prefix_len + body_len;
    if (total >= (int)sizeof(msg) - 1) {
        total = (int)sizeof(msg) - 2;
    }
    msg[total] = '\n';
    total += 1;

    uint32_t msg_len = (uint32_t)total;
    uint32_t entry_size = 4 + (uint32_t)sizeof(FILE*) + msg_len;

    __RLogState* s = &__rlog_state;
    pthread_mutex_lock(&s->mutex);

    // Lazy-open: find or create file entry while holding the mutex
    FILE* fp = NULL;
    for (uint32_t i = 0; i < __rlog_file_count; i++) {
        if (strncmp(__rlog_files[i].path, path, 255) == 0) {
            fp = __rlog_files[i].fp;
            break;
        }
    }
    if (fp == NULL && __rlog_file_count < RLOG_MAX_FILES) {
        fp = fopen(path, "a");
        if (fp != NULL) {
            strncpy(__rlog_files[__rlog_file_count].path, path, 255);
            __rlog_files[__rlog_file_count].path[255] = '\0';
            __rlog_files[__rlog_file_count].fp = fp;
            __rlog_file_count++;
        }
    }

    uint32_t space_at_end = s->capacity - s->write_pos;

    if (fp == NULL) {
        pthread_mutex_unlock(&s->mutex);
        goto fatal_check;
    }

    if (space_at_end < entry_size) {
        if (s->data_len + space_at_end + entry_size > s->capacity) {
            pthread_mutex_unlock(&s->mutex);
            goto fatal_check;
        }
        if (space_at_end >= 4) {
            uint32_t sentinel = UINT32_MAX;
            memcpy(s->buf + s->write_pos, &sentinel, 4);
        }
        s->data_len += space_at_end;
        s->write_pos = 0;
    }

    if (s->data_len + entry_size > s->capacity) {
        pthread_mutex_unlock(&s->mutex);
        goto fatal_check;
    }

    {
        uint32_t raw_len = msg_len | (1u << 31);
        memcpy(s->buf + s->write_pos, &raw_len, 4);
        memcpy(s->buf + s->write_pos + 4, &fp, sizeof(FILE*));
        memcpy(s->buf + s->write_pos + 4 + sizeof(FILE*), msg, msg_len);
        s->write_pos += entry_size;
        if (s->write_pos == s->capacity)
            s->write_pos = 0;
        s->data_len += entry_size;
    }

    pthread_cond_signal(&s->cond);
    pthread_mutex_unlock(&s->mutex);

fatal_check:
    if (level == LL_FATAL) {
        exit(__FATAL_EXIT);
    }
}

#endif // RLOG_IMPLEMENTATION

#define RLOG(level, fmt, ...) __Log_impl(level, __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define RLOG_FILE(path, level, fmt, ...) __Log_file_impl(path, level, __FILE__, __LINE__, fmt, ##__VA_ARGS__)

#ifdef __cplusplus
}
#endif

#endif // RLOG_H
