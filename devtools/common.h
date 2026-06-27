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

#define __MAX__(a, b)                                                                                                                                \
    ({                                                                                                                                               \
        __typeof__(a) _a = (a);                                                                                                                      \
        __typeof__(b) _b = (b);                                                                                                                      \
        _a > _b ? _a : _b;                                                                                                                           \
    })

#define __MIN__(a, b)                                                                                                                                \
    ({                                                                                                                                               \
        __typeof__(a) _a = (a);                                                                                                                      \
        __typeof__(b) _b = (b);                                                                                                                      \
        _a < _b ? _a : _b;                                                                                                                           \
    })

#define __STRINGIFY__(x) #x

#endif // COMMON_H
