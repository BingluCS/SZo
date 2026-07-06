//
// Created by Kai Zhao on 9/1/20.
//

#ifndef SZo_INTERPOLATORS_HPP
#define SZo_INTERPOLATORS_HPP

#include <cmath>
#include <type_traits>
#include "SZo/def.hpp"

// Bit-exactness across scalar / AVX2 / SVE2 requires the prediction helpers to NOT be
// auto-fused into FMAs (which happens by default on ARM with -ffp-contract=fast but not
// on plain x86). Fused ops are expressed explicitly with std::fma; everything else must
// stay unfused. This does NOT affect the SIMD kernels (they use explicit intrinsics).
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

namespace SZo {

template <class T>
ALWAYS_INLINE T interp_linear(T a, T b) {
    return (a + b) / 2;
}

// Floating branches below are written to match the SIMD (AVX2/SVE2) inline op
// sequence exactly; integral branches mirror SZ3's integer polynomial formulas.
template <class T>
ALWAYS_INLINE T interp_linear1(T a, T b) {
    if constexpr (std::is_integral_v<T>) {
        return (-a + 3 * b) / 2;
    }
    // 1.5*b - 0.5*a   (SVE: svmls_n(1.5*b, a, 0.5))
    return std::fma(-T(0.5), a, T(1.5) * b);
}

template <class T>
ALWAYS_INLINE T interp_quad_1(T a, T b, T c) {
    if constexpr (std::is_integral_v<T>) {
        return (3 * a + 6 * b - c) / 8;
    }
    // (3a + 6b - c) * 0.125   (SVE: svmla_n(svnmls_n(c,b,6), a, 3) * 0.125)
    T t = std::fma(T(6), b, -c);   // 6*b - c
    t = std::fma(T(3), a, t);      // 3*a + (6*b - c)
    return t * T(0.125);
}

template <class T>
ALWAYS_INLINE T interp_quad_2(T a, T b, T c) {
    if constexpr (std::is_integral_v<T>) {
        return (-a + 6 * b + 3 * c) / 8;
    }
    // (-a + 6b + 3c) * 0.125   (SVE: svmla_n(svnmls_n(a,b,6), c, 3) * 0.125)
    T t = std::fma(T(6), b, -a);   // 6*b - a
    t = std::fma(T(3), c, t);      // 3*c + (6*b - a)
    return t * T(0.125);
}

template <class T>
ALWAYS_INLINE T interp_quad_3(T a, T b, T c) {
    return (3 * a - 10 * b + 15 * c) / 8;
}

template <class T>
ALWAYS_INLINE T interp_cubic(T a, T b, T c, T d) {
    if constexpr (std::is_integral_v<T>) {
        return (-a + 9 * b + 9 * c - d) / 16;
    }
    // ((b+c)*9 - a - d) * 0.0625   (SVE/AVX2 inline order; no FMA)
    T t = (b + c) * T(9);
    t = t - a;
    t = t - d;
    return t * T(0.0625);
}

template <class T>
ALWAYS_INLINE T interp_cubic_natural(T a, T b, T c, T d) {
   return 0.575 * (b + c) - 0.075 * (a + d);
}

template<class T>
ALWAYS_INLINE T lorenzo_1d(T a, T b) {
    return 2 * b - a;
}

template<class T>
ALWAYS_INLINE T lorenzo_2d(T a, T b, T c) {
    return (b + c - a);
}

template<class T>
ALWAYS_INLINE T lorenzo_3d(T a, T b, T c, T d, T e,T f,T g) {
    return (a - b - c + d - e + f + g);
}

    


template <class T>
ALWAYS_INLINE T interp_cubic_front(T a, T b, T c, T d) {
    return (5 * a + 15 * b - 5 * c + d) / 16;
}

template <class T>
ALWAYS_INLINE T interp_cubic_front_2(T a, T b, T c, T d) {
    return (a + 6 * b - 4 * c + d) / 4;
}

template <class T>
ALWAYS_INLINE T interp_cubic_back_1(T a, T b, T c, T d) {
    return (a - 5 * b + 15 * c + 5 * d) / 16;
}

template <class T>
ALWAYS_INLINE T interp_cubic_back_2(T a, T b, T c, T d) {
    return (-5 * a + 21 * b - 35 * c + 35 * d) / 16;
}

template <class T>
ALWAYS_INLINE T interp_cubic2(T a, T b, T c, T d) {
    return (-3 * a + 23 * b + 23 * c - 3 * d) / 40;
}

template <class T>
ALWAYS_INLINE T interp_akima(T a, T b, T c, T d) {
    T t0 = 2 * b - a - c;
    T t1 = 2 * c - b - d;
    T abt0 = fabs(t0);
    T abt1 = fabs(t1);
    if (fabs(abt0 + abt1) > 1e-9) {
        return (b + c) / 2 + (t0 * abt1 + t1 * abt0) / 8 / (abt0 + abt1);
    } else {
        return (b + c) / 2;
    }
}

template <class T>
ALWAYS_INLINE T interp_pchip(T a, T b, T c, T d) {
    T pchip = (b + c) / 2;
    if ((b - a < 0) == (c - b < 0) && fabs(c - a) > 1e-9) {
        pchip += 1 / 4 * (b - a) * (c - b) / (c - a);
    }
    if ((c - b < 0) == (d - c < 0) && fabs(d - b) > 1e-9) {
        pchip -= 1 / 4 * (c - b) * (d - c) / (d - b);
    }
    return pchip;
}

}  // namespace SZo
#endif  // SZ_INTERPOLATORS_HPP
