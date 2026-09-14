// Inspired by
// https://github.com/NVIDIA/DALI/blob/main/include/dali/core/static_switch.h
// and https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h

#pragma once

#include <c10/util/Exception.h>

/// @param COND       - a boolean expression to switch by
/// @param CONST_NAME - a name given for the constexpr bool variable.
/// @param ...       - code to execute for true and false
///
/// Usage:
/// ```
/// BOOL_SWITCH(flag, BoolConst, [&] {
///     some_function<BoolConst>(...);
/// });
/// ```

#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()

#ifdef FLASHATTENTION_DISABLE_DROPOUT
  #define DROPOUT_SWITCH(COND, CONST_NAME, ...) \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;   \
    return __VA_ARGS__();                       \
  }()
#else
  #define DROPOUT_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_ALIBI
  #define ALIBI_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;   \
    return __VA_ARGS__();                       \
  }()
#else
  #define ALIBI_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_UNEVEN_K
  #define EVENK_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    constexpr static bool CONST_NAME = true;    \
    return __VA_ARGS__();                       \
  }()
#else
  #define EVENK_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_SOFTCAP
  #define SOFTCAP_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;    \
    return __VA_ARGS__();                       \
  }()
#else
  #define SOFTCAP_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_LOCAL
  #define LOCAL_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;    \
    return __VA_ARGS__();                       \
  }()
#else
  #define LOCAL_SWITCH BOOL_SWITCH
#endif

#define FP16_SWITCH(COND, ...)               \
  [&] {                                      \
    if (COND) {                              \
      using elem_type = cutlass::half_t;     \
      return __VA_ARGS__();                  \
    } else {                                 \
      using elem_type = cutlass::bfloat16_t; \
      return __VA_ARGS__();                  \
    }                                        \
  }()

#ifndef FLASHATTENTION_DISABLE_HDIM32
  #define _CASE_HDIM_32(HEADDIM, ...) if (HEADDIM <= 32) { constexpr static int kHeadDim = 32; return __VA_ARGS__(); } else
#else
  #define _CASE_HDIM_32(HEADDIM, ...) if (HEADDIM <= 32) { TORCH_CHECK(false, "FlashAttention was compiled without support for head_dim <= 32"); } else
#endif

#ifndef FLASHATTENTION_DISABLE_HDIM64
  #define _CASE_HDIM_64(HEADDIM, ...) if (HEADDIM <= 64) { constexpr static int kHeadDim = 64; return __VA_ARGS__(); } else
#else
  #define _CASE_HDIM_64(HEADDIM, ...) if (HEADDIM <= 64) { TORCH_CHECK(false, "FlashAttention was compiled without support for head_dim <= 64"); } else
#endif

#ifndef FLASHATTENTION_DISABLE_HDIM96
  #define _CASE_HDIM_96(HEADDIM, ...) if (HEADDIM <= 96) { constexpr static int kHeadDim = 96; return __VA_ARGS__(); } else
#else
  #define _CASE_HDIM_96(HEADDIM, ...) if (HEADDIM <= 96) { TORCH_CHECK(false, "FlashAttention was compiled without support for head_dim <= 96"); } else
#endif

#ifndef FLASHATTENTION_DISABLE_HDIM128
  #define _CASE_HDIM_128(HEADDIM, ...) if (HEADDIM <= 128) { constexpr static int kHeadDim = 128; return __VA_ARGS__(); } else
#else
  #define _CASE_HDIM_128(HEADDIM, ...) if (HEADDIM <= 128) { TORCH_CHECK(false, "FlashAttention was compiled without support for head_dim <= 128"); } else
#endif

#ifndef FLASHATTENTION_DISABLE_HDIM192
  #define _CASE_HDIM_192(HEADDIM, ...) if (HEADDIM <= 192) { constexpr static int kHeadDim = 192; return __VA_ARGS__(); } else
#else
  #define _CASE_HDIM_192(HEADDIM, ...) if (HEADDIM <= 192) { TORCH_CHECK(false, "FlashAttention was compiled without support for head_dim <= 192"); } else
#endif

#ifndef FLASHATTENTION_DISABLE_HDIM256
  #define _CASE_HDIM_256(HEADDIM, ...) if (HEADDIM <= 256) { constexpr static int kHeadDim = 256; return __VA_ARGS__(); } else
#else
  #define _CASE_HDIM_256(HEADDIM, ...) if (HEADDIM <= 256) { TORCH_CHECK(false, "FlashAttention was compiled without support for head_dim <= 256"); } else
#endif

#define HEADDIM_SWITCH(HEADDIM, ...)   \
  [&] {                                \
    _CASE_HDIM_32(HEADDIM, __VA_ARGS__) \
    _CASE_HDIM_64(HEADDIM, __VA_ARGS__) \
    _CASE_HDIM_96(HEADDIM, __VA_ARGS__) \
    _CASE_HDIM_128(HEADDIM, __VA_ARGS__) \
    _CASE_HDIM_192(HEADDIM, __VA_ARGS__) \
    _CASE_HDIM_256(HEADDIM, __VA_ARGS__) \
    { TORCH_CHECK(false, "Unsupported head dimension: ", HEADDIM); } \
  }()
