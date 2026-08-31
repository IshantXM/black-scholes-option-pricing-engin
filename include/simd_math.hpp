// simd_math.hpp
//
// Custom AVX2 (256-bit, 4-wide double) implementations of exp, log, erf.
// Each is built from: range reduction -> minimax polynomial (Horner's
// method, coefficients fit offline via Chebyshev regression) -> reconstruction.
//
// These exist because there is no portable AVX2 intrinsic for exp/log/erf on
// packed doubles (that requires libraries like SVML or a hand-rolled kernel
// like this one) -- calling std::exp/std::log/std::erf in a loop forces
// scalar code and kills the SIMD width advantage.

#pragma once
#include <immintrin.h>
#include <cstdint>

namespace simdmath {

// ---------------------------------------------------------------------
// exp256: vectorized exp(x) for 4 packed doubles
// ---------------------------------------------------------------------
inline __m256d exp256(__m256d x) {
    const __m256d ln2      = _mm256_set1_pd(0.6931471805599453);
    const __m256d log2e    = _mm256_set1_pd(1.4426950408889634);
    // Clamp to avoid overflow in 2^k reconstruction (double range is ~1e308)
    const __m256d max_x    = _mm256_set1_pd(708.0);
    const __m256d min_x    = _mm256_set1_pd(-708.0);
    x = _mm256_min_pd(x, max_x);
    x = _mm256_max_pd(x, min_x);

    // k = round(x / ln2)
    __m256d k = _mm256_round_pd(_mm256_mul_pd(x, log2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    // r = x - k*ln2  (single-step; ln2 is exact enough in double for this range)
    __m256d r = _mm256_fnmadd_pd(k, ln2, x);

    // Minimax polynomial for exp(r), r in [-ln2/2, ln2/2], degree 10, Horner's method.
    // Coefficients derived via Chebyshev regression against std::exp (max rel err ~1.7e-15).
    const __m256d c10 = _mm256_set1_pd(2.7307612629190347e-07);
    const __m256d c9  = _mm256_set1_pd(2.7628621208560395e-06);
    const __m256d c8  = _mm256_set1_pd(2.480242014495035e-05);
    const __m256d c7  = _mm256_set1_pd(0.00019841201828180372);
    const __m256d c6  = _mm256_set1_pd(0.0013888888011599646);
    const __m256d c5  = _mm256_set1_pd(0.008333333356105864);
    const __m256d c4  = _mm256_set1_pd(0.04166666667025099);
    const __m256d c3  = _mm256_set1_pd(0.16666666666662017);
    const __m256d c2  = _mm256_set1_pd(0.4999999999999586);
    const __m256d c1  = _mm256_set1_pd(0.9999999999999947);
    const __m256d c0  = _mm256_set1_pd(0.9999999999999996);

    __m256d p = c10;
    p = _mm256_fmadd_pd(p, r, c9);
    p = _mm256_fmadd_pd(p, r, c8);
    p = _mm256_fmadd_pd(p, r, c7);
    p = _mm256_fmadd_pd(p, r, c6);
    p = _mm256_fmadd_pd(p, r, c5);
    p = _mm256_fmadd_pd(p, r, c4);
    p = _mm256_fmadd_pd(p, r, c3);
    p = _mm256_fmadd_pd(p, r, c2);
    p = _mm256_fmadd_pd(p, r, c1);
    p = _mm256_fmadd_pd(p, r, c0);

    // Reconstruct 2^k by writing k directly into the double exponent field.
    __m128i k_i32 = _mm256_cvtpd_epi32(k);              // k as 4x int32
    __m256i k_i64 = _mm256_cvtepi32_epi64(k_i32);        // widen to 4x int64
    __m256i bias  = _mm256_set1_epi64x(1023);
    __m256i biased = _mm256_add_epi64(k_i64, bias);
    __m256i exp_bits = _mm256_slli_epi64(biased, 52);
    __m256d two_pow_k = _mm256_castsi256_pd(exp_bits);

    return _mm256_mul_pd(p, two_pow_k);
}

// ---------------------------------------------------------------------
// log256: vectorized natural log(x) for 4 packed doubles, x > 0
// ---------------------------------------------------------------------
inline __m256d log256(__m256d x) {
    const __m256d ln2 = _mm256_set1_pd(0.6931471805599453);

    __m256i xi = _mm256_castpd_si256(x);

    // Extract raw IEEE-754 exponent (11 bits) from each lane.
    __m256i shifted = _mm256_srli_epi64(xi, 52);
    // Gather the low 32 bits of each 64-bit lane (indices 0,2,4,6) into a
    // packed set of 4 int32 values occupying the low 128 bits.
    const __m256i gather_idx = _mm256_setr_epi32(0,2,4,6, 0,0,0,0);
    __m256i packed32 = _mm256_permutevar8x32_epi32(shifted, gather_idx);
    __m128i exp_i32  = _mm256_castsi256_si128(packed32);
    __m128i bias32   = _mm_set1_epi32(1023);
    __m128i e_i32    = _mm_sub_epi32(exp_i32, bias32);
    __m256d e        = _mm256_cvtepi32_pd(e_i32);         // exponent as double

    // Build mantissa in [1,2): clear the exponent field, force exponent=0 (bias 1023).
    const __m256i mantissa_mask = _mm256_set1_epi64x(0x000FFFFFFFFFFFFFLL);
    const __m256i one_exp_bits  = _mm256_set1_epi64x(0x3FF0000000000000LL);
    __m256i mant_bits = _mm256_and_si256(xi, mantissa_mask);
    mant_bits = _mm256_or_si256(mant_bits, one_exp_bits);
    __m256d m = _mm256_castsi256_pd(mant_bits);            // m in [1,2)

    // Rescale to [sqrt(0.5), sqrt(2)) so the log(m) polynomial needs fewer
    // terms (symmetric interval around 1 minimizes max error for given degree).
    const __m256d sqrt2   = _mm256_set1_pd(1.4142135623730951);
    const __m256d half    = _mm256_set1_pd(0.5);
    const __m256d one     = _mm256_set1_pd(1.0);
    __m256d too_big = _mm256_cmp_pd(m, sqrt2, _CMP_GE_OQ);
    __m256d m_adj = _mm256_blendv_pd(m, _mm256_mul_pd(m, half), too_big);
    __m256d e_adj = _mm256_blendv_pd(e, _mm256_add_pd(e, one), too_big);

    __m256d u = _mm256_sub_pd(m_adj, one);  // u in [-0.2929, 0.4142], log(x)=log1p(u)

    // Minimax polynomial for log1p(u), degree 16, Horner's method.
    // Coefficients derived via Chebyshev regression against std::log1p (max abs err ~1.6e-14).
    const __m256d c16 = _mm256_set1_pd(-0.0390530147698112);
    const __m256d c15 = _mm256_set1_pd(0.08040418062348557);
    const __m256d c14 = _mm256_set1_pd(-0.08340877322821098);
    const __m256d c13 = _mm256_set1_pd(0.07688940881205253);
    const __m256d c12 = _mm256_set1_pd(-0.081527326755039);
    const __m256d c11 = _mm256_set1_pd(0.09069663564869178);
    const __m256d c10 = _mm256_set1_pd(-0.10013298538921066);
    const __m256d c9  = _mm256_set1_pd(0.11113341191290578);
    const __m256d c8  = _mm256_set1_pd(-0.1249946845354598);
    const __m256d c7  = _mm256_set1_pd(0.14285614360857457);
    const __m256d c6  = _mm256_set1_pd(-0.16666678226831316);
    const __m256d c5  = _mm256_set1_pd(0.20000002131355785);
    const __m256d c4  = _mm256_set1_pd(-0.24999999873199905);
    const __m256d c3  = _mm256_set1_pd(0.3333333331395836);
    const __m256d c2  = _mm256_set1_pd(-0.5000000000056335);
    const __m256d c1  = _mm256_set1_pd(1.000000000000502);
    const __m256d c0  = _mm256_set1_pd(4.343747583845925e-15);
    // Coefficients derived via scripts/derive_coeffs.py (Chebyshev regression
    // of log1p(u) on u in [sqrt(0.5)-1, sqrt(2)-1], degree 16, converted to
    // the monomial/Horner basis; max abs error vs std::log1p ~1.6e-14).

    __m256d p = c16;
    p = _mm256_fmadd_pd(p, u, c15);
    p = _mm256_fmadd_pd(p, u, c14);
    p = _mm256_fmadd_pd(p, u, c13);
    p = _mm256_fmadd_pd(p, u, c12);
    p = _mm256_fmadd_pd(p, u, c11);
    p = _mm256_fmadd_pd(p, u, c10);
    p = _mm256_fmadd_pd(p, u, c9);
    p = _mm256_fmadd_pd(p, u, c8);
    p = _mm256_fmadd_pd(p, u, c7);
    p = _mm256_fmadd_pd(p, u, c6);
    p = _mm256_fmadd_pd(p, u, c5);
    p = _mm256_fmadd_pd(p, u, c4);
    p = _mm256_fmadd_pd(p, u, c3);
    p = _mm256_fmadd_pd(p, u, c2);
    p = _mm256_fmadd_pd(p, u, c1);
    p = _mm256_fmadd_pd(p, u, c0);

    return _mm256_fmadd_pd(e_adj, ln2, p);
}

// ---------------------------------------------------------------------
// erf256: vectorized erf(x) for 4 packed doubles (any sign)
// Implemented as erf(x) = sign(x) * (1 - g(|x|)*exp(-x^2)), where
// g(x) = erfc(x)*exp(x^2) is smooth & bounded on [0,6] -- fitting g()
// directly (rather than erf/erfc) avoids the ill-conditioning that a
// direct high-degree polynomial fit of erf/erfc hits on a wide range.
// ---------------------------------------------------------------------
inline __m256d erf256(__m256d x) {
    __m256d sign_mask = _mm256_set1_pd(-0.0);
    __m256d sign = _mm256_and_pd(x, sign_mask);
    __m256d ax = _mm256_andnot_pd(sign_mask, x);   // |x|
    __m256d clamped = _mm256_min_pd(ax, _mm256_set1_pd(6.0));

    // g(t) = erfc(t)*exp(t^2), degree-24 minimax polynomial, Horner's method.
    // Coefficients derived via Chebyshev regression (max abs err in erf ~8.3e-12).
    const __m256d c24 = _mm256_set1_pd(6.179935016339379e-16);
    const __m256d c23 = _mm256_set1_pd(-4.7693652998635714e-14);
    const __m256d c22 = _mm256_set1_pd(1.7387213870000922e-12);
    const __m256d c21 = _mm256_set1_pd(-3.984595393648231e-11);
    const __m256d c20 = _mm256_set1_pd(6.445379409006682e-10);
    const __m256d c19 = _mm256_set1_pd(-7.835708878231809e-09);
    const __m256d c18 = _mm256_set1_pd(7.446224498086763e-08);
    const __m256d c17 = _mm256_set1_pd(-5.682974284185786e-07);
    const __m256d c16 = _mm256_set1_pd(3.5544364784695597e-06);
    const __m256d c15 = _mm256_set1_pd(-1.8519908647449495e-05);
    const __m256d c14 = _mm256_set1_pd(8.156818793167833e-05);
    const __m256d c13 = _mm256_set1_pd(-0.0003080605532174792);
    const __m256d c12 = _mm256_set1_pd(0.0010128901953382597);
    const __m256d c11 = _mm256_set1_pd(-0.002947536909320519);
    const __m256d c10 = _mm256_set1_pd(0.00772438941083408);
    const __m256d c9  = _mm256_set1_pd(-0.018532592502548042);
    const __m256d c8  = _mm256_set1_pd(0.04123935222635505);
    const __m256d c7  = _mm256_set1_pd(-0.08572461340399223);
    const __m256d c6  = _mm256_set1_pd(0.16655950135025166);
    const __m256d c5  = _mm256_set1_pd(-0.30086772999780714);
    const __m256d c4  = _mm256_set1_pd(0.4999929643087626);
    const __m256d c3  = _mm256_set1_pd(-0.7522518603619978);
    const __m256d c2  = _mm256_set1_pd(0.9999999360706595);
    const __m256d c1  = _mm256_set1_pd(-1.1283791653151025);
    const __m256d c0  = _mm256_set1_pd(0.9999999999917165);
    // g(t) coefficients derived via scripts/derive_coeffs.py (Chebyshev
    // regression of erfc(t)*exp(t^2) on t in [0,6], degree 24, converted to
    // monomial/Horner basis; propagated max abs error in erf(x) ~8.3e-12).

    __m256d p = c24;
    p = _mm256_fmadd_pd(p, clamped, c23);
    p = _mm256_fmadd_pd(p, clamped, c22);
    p = _mm256_fmadd_pd(p, clamped, c21);
    p = _mm256_fmadd_pd(p, clamped, c20);
    p = _mm256_fmadd_pd(p, clamped, c19);
    p = _mm256_fmadd_pd(p, clamped, c18);
    p = _mm256_fmadd_pd(p, clamped, c17);
    p = _mm256_fmadd_pd(p, clamped, c16);
    p = _mm256_fmadd_pd(p, clamped, c15);
    p = _mm256_fmadd_pd(p, clamped, c14);
    p = _mm256_fmadd_pd(p, clamped, c13);
    p = _mm256_fmadd_pd(p, clamped, c12);
    p = _mm256_fmadd_pd(p, clamped, c11);
    p = _mm256_fmadd_pd(p, clamped, c10);
    p = _mm256_fmadd_pd(p, clamped, c9);
    p = _mm256_fmadd_pd(p, clamped, c8);
    p = _mm256_fmadd_pd(p, clamped, c7);
    p = _mm256_fmadd_pd(p, clamped, c6);
    p = _mm256_fmadd_pd(p, clamped, c5);
    p = _mm256_fmadd_pd(p, clamped, c4);
    p = _mm256_fmadd_pd(p, clamped, c3);
    p = _mm256_fmadd_pd(p, clamped, c2);
    p = _mm256_fmadd_pd(p, clamped, c1);
    p = _mm256_fmadd_pd(p, clamped, c0);

    __m256d neg_x2 = _mm256_mul_pd(_mm256_mul_pd(ax, ax), _mm256_set1_pd(-1.0));
    __m256d exp_neg_x2 = exp256(neg_x2);
    __m256d result = _mm256_fnmadd_pd(p, exp_neg_x2, _mm256_set1_pd(1.0)); // 1 - p*exp(-x^2)

    return _mm256_or_pd(result, sign); // reapply original sign
}

// Standard normal CDF: N(x) = 0.5*(1 + erf(x/sqrt(2)))
inline __m256d norm_cdf256(__m256d x) {
    const __m256d inv_sqrt2 = _mm256_set1_pd(0.7071067811865476);
    const __m256d half = _mm256_set1_pd(0.5);
    const __m256d one = _mm256_set1_pd(1.0);
    __m256d e = erf256(_mm256_mul_pd(x, inv_sqrt2));
    return _mm256_mul_pd(half, _mm256_add_pd(one, e));
}

} // namespace simdmath
