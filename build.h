#ifndef BUILD_H
#define BUILD_H

#ifdef __linux__
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif // _GNU_SOURCE
#endif // __linux__

#include <errno.h>
#include <limits.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#ifndef COMMON_H
#define COMMON_H

#include <assert.h>
#include <stdint.h>
#include <sys/types.h>

// Unsigned ints
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

#if defined(__unix__) || defined(__APPLE__) || defined(__linux__)
typedef size_t usize;
#else
typedef uint64_t usize;
#endif

// Signed ints
typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;

#if defined(__unix__) || defined(__APPLE__) || defined(__linux__)
typedef ssize_t isize;
#else
typedef int64_t isize;
#endif

// Floating point
typedef float f32;
typedef double f64;

static_assert(sizeof(f32) == 4, "Size of float must be 32-bits");
static_assert(sizeof(f64) == 8, "Size of double must be 64-bits");

#define MAX(a, b)                                                                                                                                    \
    ({                                                                                                                                               \
        __typeof__(a) _a = (a);                                                                                                                      \
        __typeof__(b) _b = (b);                                                                                                                      \
        _a > _b ? _a : _b;                                                                                                                           \
    })

#define MIN(a, b)                                                                                                                                    \
    ({                                                                                                                                               \
        __typeof__(a) _a = (a);                                                                                                                      \
        __typeof__(b) _b = (b);                                                                                                                      \
        _a < _b ? _a : _b;                                                                                                                           \
    })

#endif // COMMON_H

#define RLOG_IMPLEMENTATION

#ifndef RLOG_H
#define RLOG_H

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
void __Log_impl(LogLevel level, const char* file, u32 line, const char* fmt, ...);

#ifdef RLOG_IMPLEMENTATION

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

void __Log_impl(LogLevel level, const char* file, u32 line, const char* fmt, ...) {
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

#endif // RLOG_H

#define __EmptyStruct                                                                                                                                \
    struct {}

// Arena Allocator for general use (internal use only)
typedef struct {
        char* data;
        usize offset;
        usize capacity;
} __Arena;

static pthread_mutex_t __arena_mutex = PTHREAD_MUTEX_INITIALIZER;

static __Arena __arena = {nullptr, 0, 0};
static bool build_success = true;

void initArena(usize size) { __arena = (__Arena){.data = (char*)malloc(size), .offset = 0, .capacity = size}; }
void freeArena() { free(__arena.data); }

char* arenaAlloc(usize size) {
    pthread_mutex_lock(&__arena_mutex);
    RLOG(LL_TRACE, "Aquired Mutex");

    usize alignment = sizeof(void*);

    usize current_ptr = (usize)(__arena.data + __arena.offset);
    usize padding = 0;
    if (current_ptr % alignment != 0) {
        padding = alignment - (current_ptr % alignment);
    }

    if (__arena.data == nullptr && __arena.capacity == 0 && __arena.offset == 0) {
        pthread_mutex_unlock(&__arena_mutex);
        RLOG(LL_TRACE, "Released Mutex");
        RLOG(LL_FATAL, "Arena must be initialized before the build script can run");
    }

    if (__arena.offset + padding + size > __arena.capacity) {
        freeArena();
        pthread_mutex_unlock(&__arena_mutex);
        RLOG(LL_TRACE, "Released Mutex");
        RLOG(LL_FATAL, "Arena out of memory, use bigger arena size");
    }

    __arena.offset += padding;

    char* ret = __arena.data + __arena.offset;
    __arena.offset += size;
    pthread_mutex_unlock(&__arena_mutex);
    RLOG(LL_TRACE, "Released Mutex");
    return ret;
}

char* arenaCalloc(usize size) {
    char* ret = arenaAlloc(size);
    memset(ret, 0, size);
    return ret;
}

// Just allocate new memory because we don't care about fragmentation for this small of a program
char* arenaRealloc(char* data, usize old_size, usize size) {
    if (data == nullptr) {
        RLOG(LL_DEBUG, "data pointer is null... Creating new allocation");
        return arenaAlloc(size);
    }

    RLOG(LL_TRACE, "Checking if data pointer is part of the arena");
    pthread_mutex_lock(&__arena_mutex);
    RLOG(LL_TRACE, "Aquired Mutex");
    if (data < __arena.data || data > __arena.data + __arena.capacity) {
        freeArena();
        pthread_mutex_unlock(&__arena_mutex);
        RLOG(LL_TRACE, "Released Mutex");
        RLOG(LL_FATAL, "data pointer is not part of the arena");
    }

    if (data + old_size == __arena.data + __arena.offset) {
        // Extend in place
        __arena.offset += (size - old_size);
        pthread_mutex_unlock(&__arena_mutex);
        RLOG(LL_TRACE, "Released Mutex");
        return data;
    }
    pthread_mutex_unlock(&__arena_mutex);
    RLOG(LL_TRACE, "Released Mutex");

    char* new_data = (char*)arenaAlloc(size);
    memcpy(new_data, data, old_size);
    return new_data;
}

// --- Vector ---

#define Vector(type)                                                                                                                                 \
    struct {                                                                                                                                         \
            type* items;                                                                                                                             \
            usize len;                                                                                                                               \
            usize cap;                                                                                                                               \
    }

#define VectorPushBack(type, vector, elem)                                                                                                           \
    do {                                                                                                                                             \
        if ((vector)->len == (vector)->cap) {                                                                                                        \
            usize old_cap = (vector)->cap;                                                                                                           \
            if ((vector)->cap == 0) {                                                                                                                \
                (vector)->cap = 1;                                                                                                                   \
            } else {                                                                                                                                 \
                (vector)->cap *= 2;                                                                                                                  \
            }                                                                                                                                        \
            (vector)->items = (type*)arenaRealloc((char*)((vector)->items), old_cap * sizeof(type), (vector)->cap * sizeof(type));                   \
        }                                                                                                                                            \
        (vector)->items[(vector)->len] = elem;                                                                                                       \
        (vector)->len++;                                                                                                                             \
    } while (0)

// Util functions
// ! Internal use only
char* __expandPath(char* path) {
    char* expanded = (char*)arenaAlloc(PATH_MAX);

    if (path[0] == '/') {
        strncpy(expanded, path, PATH_MAX - 1);
        expanded[PATH_MAX - 1] = '\0';
    } else {
        char cwd[PATH_MAX];
        if (getcwd(cwd, sizeof(cwd)) == nullptr) {
            RLOG(LL_FATAL, "Failed to get the current working directory");
        }

        snprintf(expanded, PATH_MAX, "%s/%s", cwd, path);
    }

    char* resolved = (char*)arenaAlloc(PATH_MAX);
    if (realpath(expanded, resolved) != nullptr) {
        return resolved;
    }
    return expanded;
}

// no need for alloc because the absolute path should end in null terminator
// ! Internal use only
char* __extractFilename(char* absolute_path) {
    usize index = strlen(absolute_path) - 1;
    while (absolute_path[index--] != '/') {
    }

    return absolute_path + index + 1;
}

// ! Internal use only
char* __removeFilenameExt(char* filename) {
    usize index = strlen(filename) - 1;
    while (index > 0 && filename[index] != '.') {
        index--;
    }

    char* extended = (char*)arenaCalloc((index + 1) * sizeof(char));
    memcpy(extended, filename, index);
    return extended;
}

// ! Internal use only
char* __replaceFileExt(char* path, const char* new_ext) {
    char* dot = strrchr(path, '.'); // Find last '.'
    usize base_len = dot ? (dot - path) : strlen(path);

    usize ext_len = strlen(new_ext);
    char* ret = (char*)arenaCalloc(base_len + 1 + ext_len + 1);

    memcpy(ret, path, base_len);
    ret[base_len] = '.';
    strcpy(ret + base_len + 1, new_ext);

    return ret;
}

// ! Internal use only
char* __removeFilename(char* path) {
    usize len = strlen(path);
    while (len > 0 && path[len] != '/') {
        len--;
    }

    char* ret = (char*)arenaCalloc(len + 1);
    memcpy(ret, path, len);
    return ret;
}

// ! Internal use only
char** __splitWhitespace(char* str, usize* out_count) {
    usize cap = 8;
    usize len = 0;
    char** result = (char**)arenaAlloc(cap * sizeof(char*));

    const char* ptr = str;
    while (*ptr) {
        while (*ptr && (*ptr == ' ' || *ptr == '\t')) {
            ptr++;
        }

        if (!*ptr) {
            break;
        }

        const char* start = ptr;
        while (*ptr && *ptr != ' ' && *ptr != '\t') {
            ptr++;
        }

        usize word_len = ptr - start;
        if (word_len > 0) {
            if (len == cap) {
                result = (char**)arenaRealloc((char*)result, cap * sizeof(char*), cap * sizeof(char*) * 2);
                cap *= 2;
            }

            result[len] = (char*)arenaCalloc(word_len + 1);
            memcpy(result[len], start, word_len);
            len++;
        }
    }

    *out_count = len;
    return result;
}

typedef struct {
        char* dep_name;
        char** deps;
        usize dep_count;
} __DependencyList;

// Command Interface
typedef struct {
        char** command_chain;
        char* exec_dir;
        usize _chain_offset;
        usize _chain_capcity;
} Command;

#define COUNT_STR_ARGS(...) (sizeof((char*[]){__VA_ARGS__}) / sizeof(char*))

// ! Internal use only
Command _newCommand_impl(usize count, ...) {
    Command cmd = {0};

    va_list args;
    va_start(args, count);

    cmd._chain_capcity = count;
    cmd._chain_offset = count;
    cmd.command_chain = (char**)arenaAlloc(count * sizeof(char*));

    for (u32 i = 0; i < count; i++) {
        cmd.command_chain[i] = va_arg(args, char*);
    }

    va_end(args);

    cmd.exec_dir = nullptr;

    return cmd;
}
#define newCommand(...) _newCommand_impl(COUNT_STR_ARGS(__VA_ARGS__), __VA_ARGS__)

// ! Internal use only
Command _newCommandFromDir_impl(char* exec_dir, usize count, ...) {
    Command cmd = {0};

    va_list args;
    va_start(args, count);

    cmd._chain_capcity = count + 3;
    cmd._chain_offset = count + 3;
    cmd.command_chain = (char**)arenaAlloc((count + 3) * sizeof(char*));
    cmd.exec_dir = __expandPath(exec_dir);

    cmd.command_chain[0] = "cd";
    cmd.command_chain[1] = cmd.exec_dir;
    cmd.command_chain[2] = "&&";

    for (u32 i = 3; i < count + 3; i++) {
        cmd.command_chain[i] = va_arg(args, char*);
    }

    va_end(args);

    return cmd;
}
#define newCommandFromDir(exec_dir, ...) _newCommandFromDir_impl(exec_dir, COUNT_STR_ARGS(__VA_ARGS__), __VA_ARGS__)

void cmdPushBack(Command* cmd, char* chain) {
    if (cmd->_chain_offset + 1 > cmd->_chain_capcity) {
        cmd->command_chain =
            (char**)arenaRealloc((char*)cmd->command_chain, cmd->_chain_capcity * sizeof(char*), cmd->_chain_capcity * 2 * sizeof(char*));
        cmd->_chain_capcity *= 2;
    }

    cmd->command_chain[cmd->_chain_offset++] = chain;
}

// ! Internal use only
usize __getTerminalWidth() {
    struct winsize w;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == -1) {
        return 80; // Default fallback
    }

    return w.ws_col;
}

// ! Internal use only
usize __cmdlen(Command* cmd) {
    usize cmd_len = 0;
    for (usize i = 0; i < cmd->_chain_offset; i++) {
        if (cmd->command_chain[i] != nullptr) {
            cmd_len += strlen(cmd->command_chain[i]) + 1; // +1 For space
        }
    }

    return cmd_len + 1; // +1 For null terminator
}

// ! Internal use only
void __cmdSnprint(Command* cmd, usize size, char* buffer) {
    buffer[0] = '\0';
    usize buffer_offset = 0;
    for (usize i = 0; i < cmd->_chain_offset; i++) {
        if (cmd->command_chain[i] != nullptr) {
            usize chain_size = strlen(cmd->command_chain[i]);
            if (buffer_offset + chain_size + 1 > size - 1) {
                RLOG(LL_FATAL, "Buffer overflow");
            }

            strcpy(buffer + buffer_offset, cmd->command_chain[i]);
            // strcat(buffer, " ");
            buffer_offset += chain_size;
            buffer[buffer_offset++] = ' ';
            buffer[buffer_offset] = '\0';
        }
    }
}

void cmdPrint(Command* cmd) {
    RLOG(LL_DEBUG, "cmdPrint: command_chain=%p, _chain_offset=%zu", (void*)cmd->command_chain, cmd->_chain_offset);
    // Just don't print paths longer than PATH_MAX
    char command_buffer[PATH_MAX] = {0};
    usize cols = __getTerminalWidth();

    // print the full path if we are verbose logging
    if (__log_verbose) {
        usize cmd_len = __cmdlen(cmd);
        char* command = (char*)arenaAlloc(cmd_len);
        __cmdSnprint(cmd, cmd_len, command);
        RLOG(LL_TRACE, "%s", command);
    }

    usize cmd_buf_offset = 0;
    for (usize i = 0; i < cmd->_chain_offset; i++) {
        if (cmd->command_chain[i]) {
            usize chain_len = strlen(cmd->command_chain[i]);
            if (cmd_buf_offset + chain_len > cols - 10 /* for elipses and log val */) {
                strcat(command_buffer, "...");
                break;
            }

            strcat(command_buffer, cmd->command_chain[i]);
            strcat(command_buffer, " ");
            cmd_buf_offset += chain_len + 1;
        }
    }

    RLOG(LL_INFO, "%s", command_buffer);
}

// ! Internal use only
struct __cmd_exec_output {
        char* captured;
        u32 exit_code;
} __cmdExecAndCapture(Command* cmd) {
    if (cmd->_chain_offset == 0) {
        RLOG(LL_FATAL, "Command chain is empty");
    }

    struct __cmd_exec_output out = {0};

    // char cwd[PATH_MAX];
    // getcwd(cwd, sizeof(cwd));

    // if (cmd->exec_dir) {
    //     chdir(cmd->exec_dir);
    // }

    usize cmd_len = __cmdlen(cmd);
    const char* capture_stderr = " 2>&1";
    usize combined_len = cmd_len * sizeof(char) + strlen(capture_stderr) + 1;
    char* command = (char*)arenaAlloc(combined_len);
    RLOG(LL_TRACE, "Running command: %s", command);
    __cmdSnprint(cmd, combined_len, command);
    strcat(command, capture_stderr);

    usize captured_size = 128 * sizeof(char);
    out.captured = (char*)arenaCalloc(captured_size); // DEFAULT 128
    FILE* pipe = popen(command, "r");
    {
        if (!pipe) {
            RLOG(LL_FATAL, "Failed to open pipe");
        }

        usize result_len = 0;
        char buffer[128];

        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            size_t buf_len = strlen(buffer);
            out.captured = (char*)arenaRealloc(out.captured, captured_size, (result_len + buf_len + 1) * sizeof(char));
            captured_size = (result_len + buf_len + 1) * sizeof(char);
            strcat(out.captured, buffer);
            result_len += buf_len;
        }
    }
    u32 status = pclose(pipe);
    if (WIFEXITED(status)) {
        out.exit_code = WEXITSTATUS(status);
    } else {
        RLOG(LL_FATAL, "Command execution failed");
    }

    // if (cmd->exec_dir) {
    //     chdir(cwd);
    // }

    return out;
}

u32 cmdExec(Command* cmd) {
    if (cmd->_chain_offset == 0) {
        RLOG(LL_FATAL, "Command chain is empty");
    }

    // char cwd[PATH_MAX];
    // getcwd(cwd, sizeof(cwd));

    // if (cmd->exec_dir) {
    //     chdir(cmd->exec_dir);
    // }

    // usize command_size = strlen(cmd->command_chain[0]) + 1;
    usize command_size = __cmdlen(cmd);
    char* command = (char*)arenaCalloc(command_size);
    usize command_offset = 0;

    for (usize i = 0; i < cmd->_chain_offset; i++) {
        usize chain_size = strlen(cmd->command_chain[i]) + 1;
        if (command_offset + chain_size > command_size) {
            command = (char*)arenaRealloc(command, command_size * sizeof(char), command_size * 2 * sizeof(char));
            command_size *= 2;
        }

        strcat(command, cmd->command_chain[i]);
        strcat(command, " ");
        command_offset += chain_size;
    }

    u32 result = system(command);

    // if (cmd->exec_dir) {
    //     chdir(cwd);
    // }

    return result;
}

// Compile Commands
typedef struct {
        char* src_cmd;
        char* dir;
        char* file;
} CompileCommand;

CompileCommand newCompileCommand(Command* src_cmd, char* filepath) {
    CompileCommand comp_cmd = {0};

    usize cmd_len = __cmdlen(src_cmd);
    comp_cmd.src_cmd = (char*)arenaCalloc(cmd_len);
    RLOG(LL_TRACE, "Running __cmdSpnprint");
    __cmdSnprint(src_cmd, cmd_len, comp_cmd.src_cmd);

    comp_cmd.dir = (char*)arenaCalloc(PATH_MAX);
    getcwd(comp_cmd.dir, PATH_MAX);
    comp_cmd.file = __expandPath(filepath);

    return comp_cmd;
}

// ! Internal use only
void __compCmdAddToFile(CompileCommand* comp_cmd, FILE* file) {
    fprintf(file, "\t{\n");

    fprintf(file, "\t\t\"directory\": \"%s\",\n", comp_cmd->dir);
    fprintf(file, "\t\t\"command\": \"%s\",\n", comp_cmd->src_cmd);
    fprintf(file, "\t\t\"file\": \"%s\"\n", comp_cmd->file);

    fprintf(file, "\t}");
}

// Includes
typedef struct {
        enum {
            Include_Direct,
            Include_Symbolic,
        } type;

        char* include_path;

        union {
                __EmptyStruct direct;
                struct {
                        char* symbolic_dir;
                } symbolic;
        } include;
} Include;

Include newDirectInclude(const char* path) {
    Include inc = {0};

    inc.type = Include_Direct;
    inc.include_path = __expandPath((char*)path);

    return inc;
}

Include newSymbolicInclude(const char* path, const char* symbolic_dir) {
    Include inc = {0};

    inc.type = Include_Symbolic;
    inc.include_path = __expandPath((char*)path);
    inc.include.symbolic.symbolic_dir = (char*)arenaCalloc(strlen(symbolic_dir));
    memcpy(inc.include.symbolic.symbolic_dir, symbolic_dir, strlen(symbolic_dir));

    return inc;
}

// ! Internal use only
char* __includePath(Include* inc, const char* symlinks_path) {
    switch (inc->type) {
        case Include_Direct:
            return inc->include_path;
        case Include_Symbolic:
            char* symlink_path_expanded = (char*)arenaCalloc(PATH_MAX);
            snprintf(symlink_path_expanded, PATH_MAX, "%s/%s", __expandPath((char*)symlinks_path), inc->include.symbolic.symbolic_dir);

            // Create the symbolic dir if it doesn't exist
            // Check if symlink exists, if not create it
            struct stat st;
            if (stat(symlink_path_expanded, &st) != 0) {
                Command create_sym_link = newCommand("ln", "-sfn");
                cmdPushBack(&create_sym_link, inc->include_path);
                cmdPushBack(&create_sym_link, symlink_path_expanded);
                cmdPrint(&create_sym_link);
                cmdExec(&create_sym_link);
            }

            return symlink_path_expanded;
    }

    RLOG(LL_FATAL, "Invalid Include Type");
}

// Objects
typedef struct {
        char* src_path;
        char* filename;
        char* filename_no_ext;
        char* __link_path;
} Object;

Object newObject(char* path) {
    Object obj = {0};

    obj.src_path = __expandPath(path);
    obj.filename = __extractFilename(obj.src_path);
    obj.filename_no_ext = __removeFilenameExt(obj.filename);

    return obj;
}

// ! Internal use only
__DependencyList __objListDeps(Object* obj, char* compiler, char* flags, char* includes) {
    Command dep_cmd = newCommand(compiler, flags, includes, "-MM", obj->src_path);
    struct __cmd_exec_output cmd_out = __cmdExecAndCapture(&dep_cmd);

    char* dst = cmd_out.captured;
    for (char* src = cmd_out.captured; *src; src++) {
        if (*src != '\\' && *src != '\n') {
            *dst++ = *src;
        }
    }

    *dst = '\0';

    char* colon = strchr(cmd_out.captured, ':');
    if (!colon) {
        RLOG(LL_FATAL, "Failed to extract dependency list from:\n%s", cmd_out.captured);
    }

    *colon = '\0';
    char* dep_name = cmd_out.captured;

    usize dep_count;
    char** deps = __splitWhitespace(colon + 1, &dep_count);

    return (__DependencyList){
        .dep_name = dep_name,
        .deps = deps,
        .dep_count = dep_count,
    };
}

// ! Internal use only
char** __objBuildCommand(Object* obj, char* build_dir) {
    char* relative_path = nullptr;
    char cwd[PATH_MAX] = {0};

    getcwd(cwd, PATH_MAX);
    usize offset = 0;
    while (cwd[offset] == obj->src_path[offset]) {
        offset++;
    }
    relative_path = obj->src_path + offset + 1;

    char* expanded_build_dir = __expandPath(build_dir);
    char* new_path = (char*)arenaAlloc(PATH_MAX);
    RLOG(LL_TRACE, "New Path: %s", new_path);
    snprintf(new_path, PATH_MAX, "%s/%s", expanded_build_dir, relative_path);
    RLOG(LL_TRACE, "New Path: %s", new_path);

    char* filename_removed = __removeFilename(new_path);
    RLOG(LL_TRACE, "Filename removed: %s", filename_removed);
    Command make_path = newCommand("mkdir", "-p", filename_removed);
    cmdPrint(&make_path);
    u32 result = cmdExec(&make_path);
    if (result != 0) {
        RLOG(LL_FATAL, "Failed to create directory: %s", filename_removed);
    }

    new_path = __replaceFileExt(new_path, "o");
    obj->__link_path = new_path;

    char** output = (char**)arenaCalloc(4 * sizeof(char*));

    output[0] = "-c";
    output[1] = relative_path;
    output[2] = "-o";
    output[3] = new_path;

    return output;
}

typedef Vector(char*) __StrVec;
typedef Vector(CompileCommand) __CompCmdVec;

// ! Internal use only
void __objBuild(
    Object* obj,
    char* compiler,
    char* flags,
    char* includes,
    char* build_dir,
    __StrVec* obj_files,
    pthread_mutex_t* obj_files_mutex,
    __CompCmdVec* comp_cmds,
    pthread_mutex_t* comp_cmd_mutex
) {
    __DependencyList deps_list = __objListDeps(obj, compiler, flags, includes);

    Command cmd = newCommand(compiler, flags, includes);
    char** build_command = __objBuildCommand(obj, build_dir);

    for (u32 i = 0; i < 4; i++) {
        cmdPushBack(&cmd, build_command[i]);
    }

    // Add the a new compile command to the list of compile commands
    pthread_mutex_lock(comp_cmd_mutex);
    VectorPushBack(CompileCommand, comp_cmds, newCompileCommand(&cmd, obj->src_path));
    pthread_mutex_unlock(comp_cmd_mutex);

    // Check if the object file needs a rebuild
    bool needs_rebuild = true;
    struct stat attr;

    if (stat(build_command[3], &attr) == 0) {
        time_t object_time = attr.st_mtime;
        needs_rebuild = false;

        struct stat dep_attr;
        for (int i = 0; i < deps_list.dep_count; i++) {
            char* dep_to_check = deps_list.deps[i];

            if (stat(dep_to_check, &dep_attr) == 0) {
                time_t dep_time = dep_attr.st_mtime + 1; // Add a second in case it was just saved
                if (dep_time > object_time) {
                    needs_rebuild = true;
                    break;
                }
            } else {
                needs_rebuild = true;
                break;
            }
        }
    }

    pthread_mutex_lock(obj_files_mutex);
    VectorPushBack(char*, obj_files, build_command[3]);
    pthread_mutex_unlock(obj_files_mutex);

    if (needs_rebuild) {
        cmdPrint(&cmd);
        u32 ret_code = cmdExec(&cmd);

        if (ret_code != 0) {
            build_success = false;
            RLOG(LL_FATAL, "Object file build %s failed with code %d", build_command[3], ret_code);
        }
    } else {
        RLOG(LL_DEBUG, "Skipping Object file build %s", build_command[3]);
    }
}

// Object queue

// ! Internal use only
typedef struct {
        Object** objects;
        Object** start;
        Object** end;
        usize cap;
} __ObjectQueue;

// ! Internal use only
__ObjectQueue __newObjectQueue(usize cap) {
    __ObjectQueue queue = {0};

    queue.objects = (Object**)arenaAlloc(cap * sizeof(Object*));
    queue.start = queue.objects;
    queue.end = queue.objects;
    queue.cap = cap;

    return queue;
}

// ! Internal use only
void __objQPush(__ObjectQueue* q, Object* obj) {
    Object** next_end = q->end + 1;
    if (next_end >= q->objects + q->cap) {
        next_end = q->objects;
    }

    if (next_end == q->start) {
    build_h___objQPush__obj_q_push_rerun:
        usize start_offset = q->start - q->objects;
        usize end_offset = q->end - q->objects;

        q->objects = (Object**)arenaRealloc((char*)q->objects, q->cap * sizeof(Object*), q->cap * 2 * sizeof(Object*));
        q->cap *= 2;

        // Put the pointers back
        // q->start = q->objects + (q->start - q->objects);
        // q->end = q->objects + (q->end - q->objects);
        q->start = q->objects + start_offset;
        q->end = q->objects + end_offset;

        next_end = q->end + 1;
        if (next_end >= q->objects + q->cap) {
            goto build_h___objQPush__obj_q_push_rerun;
        }
    }

    *q->end = obj;
    q->end = next_end;
}

// ! Internal use only
Object* __objQPop(__ObjectQueue* q) {
    if (q->start == q->end) {
        return nullptr;
    }

    Object* obj = *q->start;
    q->start++;
    if (q->start >= q->objects + q->cap) {
        q->start = q->objects;
    }
    return obj;
}

// Link
typedef struct {
        enum {
            Link_Direct,
            Link_Path,
            Link_Framework,
        } type;
        char* dep_name;
        union {
                __EmptyStruct direct_link;
                __EmptyStruct framework;
                struct {
                        char* dirname;
                        /* optional */ char* direct_path;
                } path_link;
        } link;
} Link;

Link newDirectLink(char* dep_name) {
    Link link = {0};

    link.type = Link_Direct;
    link.dep_name = dep_name;

    return link;
}

Link newPathLink(char* path) {
    Link link = {0};

    link.type = Link_Path;
    link.link.path_link.dirname = nullptr;
    link.link.path_link.direct_path = __expandPath(path);
    link.dep_name = nullptr;

    return link;
}

Link newPathLinkWithDep(char* dep_name, char* dir_name) {
    Link link = {0};

    link.type = Link_Path;
    link.dep_name = dep_name;
    link.link.path_link.dirname = __expandPath(dir_name);
    link.link.path_link.direct_path = nullptr;

    return link;
}

Link newFramework(char* dep_name) {
    Link link = {0};

    link.type = Link_Framework;
    link.dep_name = dep_name;

    return link;
}

char* __linkable(Link* link) {
    constexpr usize MAX_LEN = PATH_MAX * 2; // x2 for safety
    char* result = (char*)arenaCalloc(MAX_LEN);
    switch (link->type) {
        case Link_Direct:
            snprintf(result, MAX_LEN, "-l%s", link->dep_name);
            break;
        case Link_Path:
            if (link->link.path_link.direct_path) {
                return link->link.path_link.direct_path;
            }

            snprintf(result, MAX_LEN, "-L%s -Wl,-rpath,%s -l%s", link->link.path_link.dirname, link->link.path_link.dirname, link->dep_name);
            break;
        case Link_Framework:
            snprintf(result, MAX_LEN, "-framework %s", link->dep_name);
            break;
    }

    return result;
}

// Build

typedef Vector(Command) __CmdVec;

typedef Vector(Include) __IncVec;

typedef Vector(Object) __ObjVec;

typedef struct {
        usize step;
        usize object;
} LinkedObject;

typedef Vector(LinkedObject) __LinkedObjVec;

typedef Vector(Link) __LinkVec;

typedef struct {
        char* compiler;
        char* output_file;
        __CmdVec pre_step_commands;
        __IncVec includes;
        __ObjVec objects;
        __LinkedObjVec linked_objects;
        __LinkVec links;
        __StrVec comp_flags;
        __StrVec link_flags;
        bool skip_linking;
} __BuildStep;

typedef Vector(__BuildStep) __BuildStepVec;

typedef struct {
        pthread_mutex_t mutex;

        atomic_bool all_jobs_queued;
        atomic_bool all_jobs_complete;

        __ObjectQueue objects;
} __BuildJob;

typedef Vector(__BuildJob) __BuildJobVec;

typedef struct {
        enum {
            Os_Invalid,
            Os_MacOS,
            Os_Linux,
            Os_Windows,
        } os;

        enum {
            Mode_Debug,
            Mode_Release,
        } mode;
} Builtin;

typedef struct {
        Builtin builtin;

        // ! Internal use only
        char* __build_dir;
        char* __symlink_dir;
        char* __default_compiler;

        __BuildStepVec __build_steps;
        __BuildJobVec __build_jobs;
        usize __jobs;

        bool __skip_compile_commands;

        int __argc;
        char** __argv;
} Build;

const char* DEFAULT_COMPLIER = "clang";
const char* SYM_LINKS_PATH = "/sym_links";

// ! Internal Use Only
void __rebuildYourself(Build* build) {
    const char* build_c = "build.c";
    const char* build_h = "build.h";

#ifdef _WIN32
    const char* build_exe = "build.exe";
#else
    const char* build_exe = "build";
#endif

    struct stat build_c_attr;
    struct stat build_h_attr;

    if (stat(build_c, &build_c_attr) != 0 || stat(build_h, &build_h_attr) != 0) {
        RLOG(LL_FATAL, "build.c or build.h file not found");
    }

    time_t build_c_time = build_c_attr.st_mtime;
    time_t build_h_time = build_h_attr.st_mtime;

    bool rebuild = false;
    struct stat build_exe_attr;
    if (stat(build_exe, &build_exe_attr) == 0) {
        time_t build_exe_time = build_exe_attr.st_mtime;

        if (build_c_time > build_exe_time || build_h_time > build_exe_time) {
            rebuild = true;
        }
    } else {
        rebuild = true;
    }

    if (rebuild) {
        RLOG(LL_WARN, "Rebuilding build system...");

        Command cmd;
        if (build->builtin.mode == Mode_Debug) {
            cmd = newCommand(build->__default_compiler, "-std=c23", "-g", "-O0", (char*)build_c, "-o", (char*)build_exe);
        } else {
            cmd = newCommand(build->__default_compiler, "-std=c23", (char*)build_c, "-o", (char*)build_exe);
        }

        cmdPrint(&cmd);
        u32 result = cmdExec(&cmd);

        if (result != 0) {
            RLOG(LL_FATAL, "Build System rebuild failed");
        }

#ifdef __WIN32
        Command run_cmd = newCommand(".\\build.exe");
#else
        Command run_cmd = newCommand("./build");
#endif

        for (int i = 1; i < build->__argc; i++) {
            cmdPushBack(&run_cmd, build->__argv[i]);
        }

        RLOG(LL_INFO, "Rerunning Build Script...");
        result = cmdExec(&run_cmd);
        if (result != 0) {
            RLOG(LL_FATAL, "Build Script execution failed");
        }
        exit(0);
    }
}

// ! Internal Use Only
void __initCommon(Build* build) {
    build->__symlink_dir = __expandPath(build->__build_dir);
    build->__symlink_dir =
        (char*)arenaRealloc(build->__symlink_dir, strlen(build->__symlink_dir), strlen(build->__symlink_dir) + strlen(SYM_LINKS_PATH) + 1);
    strcat(build->__symlink_dir, SYM_LINKS_PATH);

    Command create_sym_link_dir = newCommand("mkdir", "-p", build->__symlink_dir);
    cmdPrint(&create_sym_link_dir);
    cmdExec(&create_sym_link_dir);

    // Create an initial build step
    VectorPushBack(__BuildStep, &build->__build_steps, (__BuildStep){0});

    __BuildStep* step = &build->__build_steps.items[build->__build_steps.len - 1];
    step->pre_step_commands = (__CmdVec){0};
    step->comp_flags = (__StrVec){0};
    step->link_flags = (__StrVec){0};
    step->links = (__LinkVec){0};
    step->includes = (__IncVec){0};
    step->objects = (__ObjVec){0};
    step->skip_linking = false;

    build->__skip_compile_commands = false;
    build->__jobs = 1;
    build->__build_jobs = (__BuildJobVec){0};

    build->builtin = (Builtin){
        .os = Os_Invalid,
        .mode = Mode_Debug,
    };

#if defined(__linux__)
    build->builtin.os = Os_Linux;
#elif defined(__APPLE__)
    build->builtin.os = Os_MacOS;
#elif defined(__WIN32)
    build->builtin.os = Os_Windows;
#elif defined(__WIN64)
    build->builtin.os = Os_Windows;
#endif

    // Argparse
    for (usize i = 0; i < build->__argc; i++) {
        char* arg = build->__argv[i];

        if (strcmp(arg, "-Release") == 0) {
            build->builtin.mode = Mode_Release;
        }

        if (strcmp(arg, "-j") == 0) {
            i++;
            errno = 0;
            arg = build->__argv[i];
            char* end;
            build->__jobs = MAX(strtol(arg, &end, 10), 1);
            RLOG(LL_DEBUG, "Number of jobs: %zu", build->__jobs);

            if (arg == end) {
                continue;
            } else if (errno == ERANGE) {
                RLOG(LL_FATAL, "%s is not a valid number of jobs", arg);
            } else if (*end != '\0') {
                RLOG(LL_FATAL, "extra characters after the number of jobs: %s", end);
            }
        }
    }

    __rebuildYourself(build);
}

Build newBuild(const char* dir_name, int argc, char** argv) {
    initLog();
    Build build = {0};

    build.__argc = argc;
    build.__argv = argv;
    build.__build_dir = __expandPath((char*)dir_name);
    build.__default_compiler = (char*)DEFAULT_COMPLIER;

    __initCommon(&build);

    return build;
}

Build newBuildWithCompiler(const char* dir_name, char* compiler, int argc, char** argv) {
    initLog();
    Build build = {0};

    build.__argc = argc;
    build.__argv = argv;
    build.__build_dir = __expandPath((char*)dir_name);
    build.__default_compiler = compiler;

    __initCommon(&build);

    return build;
}

// Actual building functions
// ! Internal use only
void __buildExportCompileCommands(Build* build, __CompCmdVec* compile_commands) {
    if (build->__skip_compile_commands) {
        return;
    }

    FILE* compile_commands_file = fopen("compile_commands.json", "w");
    if (!compile_commands_file) {
        RLOG(LL_ERROR, "Failed to open compile_commands.json");
        return;
    }

    fprintf(compile_commands_file, "[\n");

    for (usize i = 0; i < compile_commands->len; i++) {
        __compCmdAddToFile(&compile_commands->items[i], compile_commands_file);
        if (i < compile_commands->len - 1) {
            fprintf(compile_commands_file, ",\n");
        } else {
            fprintf(compile_commands_file, "\n");
        }
    }

    fprintf(compile_commands_file, "]");
    fclose(compile_commands_file);
}

/**
 * @brief Disables the generation of compile_commands.json file during the build process.
 *
 * This function sets the internal flag to skip the export of compile commands,
 * which is useful when you don't need LSP support or want to avoid generating
 * the compile_commands.json file for performance reasons.
 *
 * @param build Pointer to the Build structure to modify
 */
void buildSkipCompileCommands(Build* build) { build->__skip_compile_commands = true; }

/**
 * @brief Adds a pre-build command to be executed before compilation begins.
 *
 * Pre-build commands are executed in the order they were added, before any
 * compilation or linking takes place. This is useful for code generation,
 * copying resources, or any other setup tasks that need to happen before
 * the main build process.
 *
 * @param build Pointer to the Build structure to modify
 * @param command The Command to execute during the pre-build phase
 */
void buildAddPrebuildCommand(Build* build, Command command) {
    size_t build_step_index = build->__build_steps.len - 1;
    __CmdVec* vec = &build->__build_steps.items[build_step_index].pre_step_commands;
    VectorPushBack(Command, vec, command);
}

/**
 * @brief Adds a source file object to the current build step for compilation.
 *
 * Objects represent source files that need to be compiled into object files.
 * The build system will automatically handle dependency tracking and only
 * recompile when necessary. Multiple objects can be added to create a
 * multi-file compilation unit.
 *
 * @param build Pointer to the Build structure to modify
 * @param obj The Object representing a source file to compile
 * @return Pointer to the Object that was added to the build step
 */
LinkedObject buildAddObject(Build* build, Object obj) {
    size_t build_step_index = build->__build_steps.len - 1;
    __ObjVec* vec = &build->__build_steps.items[build_step_index].objects;
    VectorPushBack(Object, vec, obj);
    return (LinkedObject){
        .step = build_step_index,
        .object = vec->len - 1,
    };
}

/**
 * @brief Adds an include directory or symbolic link to the current build step.
 *
 * Include paths are used by the compiler to locate header files. The build
 * system supports both direct includes (standard -I flags) and symbolic
 * includes (which create symbolic links in a managed directory structure).
 *
 * @param build Pointer to the Build structure to modify
 * @param inc The Include specifying the header search path or symbolic link
 */
void buildAddInclude(Build* build, Include inc) {
    size_t build_step_index = build->__build_steps.len - 1;
    __IncVec* vec = &build->__build_steps.items[build_step_index].includes;
    VectorPushBack(Include, vec, inc);
}

/**
 * @brief Adds a library link dependency to the current build step.
 *
 * Links specify libraries that should be linked with the final executable
 * or library. Supports various link types including direct library names,
 * path-based linking, and framework linking (on macOS). Links are processed
 * during the linking phase after all object files have been compiled.
 *
 * @param build Pointer to the Build structure to modify
 * @param link The Link specifying the library to link against
 */
void buildAddLink(Build* build, Link link) {
    size_t build_step_index = build->__build_steps.len - 1;
    __LinkVec* vec = &build->__build_steps.items[build_step_index].links;
    VectorPushBack(Link, vec, link);
}

/**
 * @brief Adds a compilation flag to the current build step.
 *
 * Compilation flags are passed directly to the compiler during the compilation
 * phase. These can include optimization flags (-O2, -O3), debug flags (-g),
 * warning flags (-Wall, -Wextra), language standard flags (-std=c23), or any
 * other compiler-specific options.
 *
 * @param build Pointer to the Build structure to modify
 * @param flag The compilation flag string to add (e.g., "-Wall", "-O2")
 */
void buildAddCompilationFlag(Build* build, char* flag) {
    size_t build_step_index = build->__build_steps.len - 1;
    __StrVec* vec = &build->__build_steps.items[build_step_index].comp_flags;
    VectorPushBack(char*, vec, flag);
}

/**
 * @brief Adds a linking flag to the current build step.
 *
 * Linking flags are passed to the compiler/linker during the linking phase
 * when creating the final executable or library. These can include linker
 * scripts, symbol export options, library search paths, or other linker-specific
 * directives that are separate from library dependencies.
 *
 * @param build Pointer to the Build structure to modify
 * @param flag The linking flag string to add (e.g., "-static", "-pie")
 */
void buildAddLinkingFlag(Build* build, char* flag) {
    size_t build_step_index = build->__build_steps.len - 1;
    __StrVec* vec = &build->__build_steps.items[build_step_index].link_flags;
    VectorPushBack(char*, vec, flag);
}

/**
 * @brief Links a previously compiled object file to the current build step.
 *
 * This function adds a compiled object file's path as a linking flag to the
 * current build step. The object must have been compiled first (have a valid
 * __link_path) before it can be linked. This is useful when you need to link
 * object files from previous build steps or when manually managing object
 * file dependencies between different parts of your build.
 *
 * The function will terminate the build with a fatal error if the object
 * has not been compiled yet (no __link_path available).
 *
 * @param build Pointer to the Build structure to modify
 * @param obj Pointer to the Object that has been compiled and should be linked
 */
void buildAddLinkedObject(Build* build, LinkedObject obj) {
    usize build_step_index = build->__build_steps.len - 1;
    __LinkedObjVec* linked_objects = &build->__build_steps.items[build_step_index].linked_objects;
    VectorPushBack(LinkedObject, linked_objects, obj);
}

/**
 * @brief Sets the compiler for the current build step.
 *
 * Overrides the default compiler (usually "clang") for this specific build step.
 * This allows different build steps to use different compilers, which can be
 * useful for cross-compilation, using different compiler versions, or when
 * building components that require specific compiler features.
 *
 * @param build Pointer to the Build structure to modify
 * @param compiler The compiler executable name or path (e.g., "gcc", "clang++")
 */
void buildStepSetCompiler(Build* build, char* compiler) {
    size_t build_step_index = build->__build_steps.len - 1;
    build->__build_steps.items[build_step_index].compiler = compiler;
}

/**
 * @brief Configures the current build step to skip the linking phase.
 *
 * When linking is skipped, only compilation will be performed, producing
 * object files without creating a final executable or library. This is
 * useful for creating object-only builds, library archives, or when you
 * want to handle linking manually in a separate step.
 *
 * @param build Pointer to the Build structure to modify
 */
void buildStepSkipLinking(Build* build) {
    size_t build_step_index = build->__build_steps.len - 1;
    build->__build_steps.items[build_step_index].skip_linking = true;
}

/**
 * @brief Sets the output file path for the current build step.
 *
 * Specifies the name and location of the final executable or library that
 * will be produced by this build step. The output file will be created in
 * the build directory. If not set, the build step will need to have linking
 * disabled or the build will fail during the linking phase.
 *
 * @param build Pointer to the Build structure to modify
 * @param output_file The output file name/path for the build step
 */
void buildStepSetOutput(Build* build, char* output_file) {
    size_t build_step_index = build->__build_steps.len - 1;
    build->__build_steps.items[build_step_index].output_file = output_file;
}

/**
 * @brief Creates a new build step in the build pipeline.
 *
 * Build steps allow you to organize complex builds into multiple phases,
 * each with their own compiler settings, flags, objects, and dependencies.
 * This is useful for building multiple executables, libraries with different
 * configurations, or when you need different compilation settings for
 * different parts of your project.
 *
 * After calling this function, subsequent build configuration calls will
 * apply to the newly created build step. The new step inherits the default
 * compiler from the Build structure but starts with empty collections for
 * all other settings.
 *
 * @param build Pointer to the Build structure to modify
 */
void buildStep(Build* build) {
    VectorPushBack(__BuildStep, &build->__build_steps, (__BuildStep){0});

    __BuildStep* step = &build->__build_steps.items[build->__build_steps.len - 1];
    step->pre_step_commands = (__CmdVec){0};
    step->comp_flags = (__StrVec){0};
    step->link_flags = (__StrVec){0};
    step->links = (__LinkVec){0};
    step->includes = (__IncVec){0};
    step->objects = (__ObjVec){0};
    step->skip_linking = false;
    step->compiler = build->__default_compiler;
}

typedef struct {
        Build* build;
        __StrVec includes;
        __BuildStep* step;
        __StrVec* obj_files;
        pthread_mutex_t* obj_mutex;
        __CompCmdVec* compile_cmds;
        pthread_mutex_t* comp_cmd_mutex;
        usize job_number;
} __CompileThreadArgs;

// ! Internal use only
void* __threadedCompile(void* arg) {
    __CompileThreadArgs* args = (__CompileThreadArgs*)arg;
    __BuildJob* build_jobs = args->build->__build_jobs.items;

    pthread_mutex_lock(&build_jobs[args->job_number].mutex);
    RLOG(LL_TRACE, "Locked jobs %d mutex", args->job_number);
    __ObjectQueue* q = &build_jobs[args->job_number].objects;

    bool q_is_empty = q->start == q->end;
    bool all_jobs_queued = build_jobs[args->job_number].all_jobs_queued;
    pthread_mutex_unlock(&build_jobs[args->job_number].mutex);
    RLOG(LL_TRACE, "Unlocked jobs %d mutex", args->job_number);

    while (!q_is_empty || !all_jobs_queued) {
        if (!q_is_empty) {
            pthread_mutex_lock(&build_jobs[args->job_number].mutex);
            RLOG(LL_TRACE, "Locked jobs %d mutex", args->job_number);
            Object* obj = __objQPop(q);
            pthread_mutex_unlock(&build_jobs[args->job_number].mutex);
            RLOG(LL_TRACE, "Unlocked jobs %d mutex", args->job_number);

            // Concatenate all the includes into 1 string
            usize includes_cap = MAX(1, PATH_MAX * args->includes.len);
            char* includes_str = (char*)arenaCalloc(PATH_MAX * args->includes.len);
            usize includes_offset = 0;
            for (usize i = 0; i < args->includes.len; i++) {
                char* inc = args->includes.items[i];
                usize inc_len = strlen(inc);
                if (includes_offset + inc_len > includes_cap) {
                    usize new_inc_cap = MAX(includes_cap * 2, includes_cap + inc_len);
                    includes_str = (char*)arenaRealloc(includes_str, includes_cap, new_inc_cap * sizeof(char));
                    includes_cap = new_inc_cap;
                }

                strcat(includes_str, inc);
                strcat(includes_str, " ");
            }

            // Concatenate all the compilation flags into 1 string
            usize flags_cap = MAX(1, PATH_MAX * args->step->comp_flags.len);
            char* comp_flags_str = (char*)arenaCalloc(flags_cap);
            usize flags_offset = 0;
            for (usize i = 0; i < args->step->comp_flags.len; i++) {
                char* flag = args->step->comp_flags.items[i];
                usize flag_len = strlen(flag);
                if (flags_offset + flag_len > flags_cap) {
                    usize new_flag_cap = MAX(flags_cap * 2, flags_cap + flag_len);
                    comp_flags_str = (char*)arenaRealloc(comp_flags_str, flags_cap, new_flag_cap * sizeof(char));
                    flags_cap = new_flag_cap;
                }

                strcat(comp_flags_str, flag);
                strcat(comp_flags_str, " ");
            }

            char* compiler_to_use;
            if (args->step->compiler) {
                compiler_to_use = args->step->compiler;
            } else {
                compiler_to_use = args->build->__default_compiler;
            }

            __objBuild(
                obj, compiler_to_use, comp_flags_str, includes_str, args->build->__build_dir, args->obj_files, args->obj_mutex, args->compile_cmds,
                args->comp_cmd_mutex
            );
        }

        pthread_mutex_lock(&build_jobs[args->job_number].mutex);
        RLOG(LL_TRACE, "Locked jobs %d mutex", args->job_number);
        q_is_empty = q->start == q->end;
        all_jobs_queued = build_jobs[args->job_number].all_jobs_queued;
        pthread_mutex_unlock(&build_jobs[args->job_number].mutex);
        RLOG(LL_TRACE, "Unlocked jobs %d mutex", args->job_number);
    }

    build_jobs[args->job_number].all_jobs_complete = true;

    return nullptr;
}

void buildBuild(Build* build) {
    build->__build_jobs = (__BuildJobVec){
        .items = (void*)arenaAlloc(sizeof(__BuildJob) * build->__jobs),
        .len = 0,
        .cap = build->__jobs,
    };

    for (usize i = 0; i < build->__jobs; i++) {
        __BuildJob job = (__BuildJob){
            .all_jobs_queued = false,
            .all_jobs_complete = false,
            .objects = __newObjectQueue(100),
        };

        VectorPushBack(__BuildJob, &build->__build_jobs, job);
        pthread_mutex_init(&build->__build_jobs.items[i].mutex, nullptr);
    }

    __CompCmdVec compile_commands = (__CompCmdVec){0};
    pthread_mutex_t comp_command_mutex = PTHREAD_MUTEX_INITIALIZER;

    for (int step_index = 0; step_index < build->__build_steps.len; step_index++) {
        RLOG(LL_INFO, "========== RUNNING BUILD STEP %d ==========", step_index + 1);
        __BuildStep* step = &build->__build_steps.items[step_index];

        // Run the prebuild commands
        for (usize i = 0; i < step->pre_step_commands.len; i++) {
            Command* cmd = &step->pre_step_commands.items[i];
            cmdPrint(cmd);
            u32 result = cmdExec(cmd);

            if (result != 0) {
                RLOG(LL_FATAL, "Error executing command");
            }
        }

        // Create the build step includes paths
        __StrVec includes = (__StrVec){0};
        for (usize i = 0; i < step->includes.len; i++) {
            Include* inc = &step->includes.items[i];

            VectorPushBack(char*, &includes, "-I");
            char* inc_path = __includePath(inc, build->__symlink_dir);
            VectorPushBack(char*, &includes, inc_path);
        }

        // Create the build step object files
        __StrVec object_files = (__StrVec){0};
        pthread_mutex_t obj_files_mutex = PTHREAD_MUTEX_INITIALIZER;

        // Create and launch the threads
        pthread_t* threads = (pthread_t*)arenaCalloc(build->__jobs * sizeof(pthread_t));
        __CompileThreadArgs* args = (__CompileThreadArgs*)arenaCalloc(build->__jobs * sizeof(__CompileThreadArgs));

        for (usize i = 0; i < build->__jobs; i++) {
            build->__build_jobs.items[i].all_jobs_queued = false;
            build->__build_jobs.items[i].all_jobs_complete = false;

            // Reset queue pointers to clear any leftover objects
            build->__build_jobs.items[i].objects.start = build->__build_jobs.items[i].objects.objects;
            build->__build_jobs.items[i].objects.end = build->__build_jobs.items[i].objects.objects;

            args[i] = (__CompileThreadArgs){
                .build = build,
                .includes = includes,
                .step = step,
                .obj_files = &object_files,
                .obj_mutex = &obj_files_mutex,
                .compile_cmds = &compile_commands,
                .comp_cmd_mutex = &comp_command_mutex,
                .job_number = i,
            };

            pthread_create(&threads[i], nullptr, __threadedCompile, &args[i]);
        }

        // Push all the work to the thread queues round robin
        usize job_index = 0;
        for (usize i = 0; i < step->objects.len; i++) {
            Object* obj = &step->objects.items[i];

            pthread_mutex_lock(&build->__build_jobs.items[job_index].mutex);
            __objQPush(&build->__build_jobs.items[job_index].objects, obj);
            pthread_mutex_unlock(&build->__build_jobs.items[job_index].mutex);

            job_index = (job_index + 1) % build->__jobs;
        }

        // After the curretn job steps have been queued then let each thread know that
        // there will be no more work
        for (usize i = 0; i < build->__jobs; i++) {
            build->__build_jobs.items[i].all_jobs_queued = true;
        }

        // Wait for all the build jobs to finish
        for (usize i = 0; i < build->__jobs; i++) {
            pthread_join(threads[i], nullptr);
        }

        if (!build_success) {
            RLOG(LL_FATAL, "Build failed");
        }

        // Run the build step linking phase
        if (step->skip_linking == false) {
            if (step->output_file == nullptr) {
                RLOG(LL_FATAL, "Output file not specified");
            }

            Command linking_cmd;
            if (step->compiler) {
                linking_cmd = newCommand(step->compiler);
            } else {
                linking_cmd = newCommand(build->__default_compiler);
            }

            if (step->link_flags.len == 0) {
                // TODO: review
                // By default use the c23 standard
                cmdPushBack(&linking_cmd, "-std=c23");
            } else {
                for (usize i = 0; i < step->link_flags.len; i++) {
                    cmdPushBack(&linking_cmd, step->link_flags.items[i]);
                }
            }

            cmdPushBack(&linking_cmd, "-o");

            char output_file[PATH_MAX];
            snprintf(output_file, PATH_MAX, "%s/%s", build->__build_dir, step->output_file);
            cmdPushBack(&linking_cmd, output_file);

            // Add all the object files
            for (usize i = 0; i < object_files.len; i++) {
                cmdPushBack(&linking_cmd, object_files.items[i]);
            }

            // Add all the linked object files
            for (usize i = 0; i < step->linked_objects.len; i++) {
                LinkedObject linked_object = step->linked_objects.items[i];

                Object* object = &build->__build_steps.items[linked_object.step].objects.items[linked_object.object];

                if (object->__link_path == nullptr) {
                    RLOG(LL_FATAL, "Object %s has no linked path... Object has not been compiled yet", object->src_path);
                }

                cmdPushBack(&linking_cmd, object->__link_path);
            }

            // Link everything that is needed
            for (usize i = 0; i < step->links.len; i++) {
                cmdPushBack(&linking_cmd, __linkable(&step->links.items[i]));
            }

            // Execute the linking command
            cmdPrint(&linking_cmd);
            u32 link_result = cmdExec(&linking_cmd);
            if (link_result != 0) {
                RLOG(LL_FATAL, "Linking failed");
            }
        }
    }

    for (usize i = 0; i < build->__jobs; i++) {
        pthread_mutex_destroy(&build->__build_jobs.items[i].mutex);
    }

    __buildExportCompileCommands(build, &compile_commands);
}

#endif // BUILD_H
