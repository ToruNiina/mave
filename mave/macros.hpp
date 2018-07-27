#ifndef MAVE_MACROS_HPP
#define MAVE_MACROS_HPP

#ifdef __GNUC__
#define MAVE_INLINE inline __attribute__((always_inline))
#elif defined(_MSC_VER)
#define MAVE_INLINE inline __forceinline
#else
#define MAVE_INLINE inline
#endif

#endif // MAVE_MACROS_HPP
