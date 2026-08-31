// black_scholes_avx2.hpp
// AVX2-vectorized Black-Scholes pricer: prices 4 options per call using
// packed doubles and the custom simd_math exp/log/erf kernels.
#pragma once
#include <immintrin.h>
#include "simd_math.hpp"

namespace bs {

// S, K, r, sigma, T are each __m256d holding 4 options' parameters.
// Writes 4 call prices to call_out and 4 put prices to put_out.
inline void price_avx2(__m256d S, __m256d K, __m256d r, __m256d sigma, __m256d T,
                        double* call_out, double* put_out) {
    __m256d half = _mm256_set1_pd(0.5);
    __m256d one  = _mm256_set1_pd(1.0);

    __m256d sqrtT = _mm256_sqrt_pd(T);
    __m256d sigma2 = _mm256_mul_pd(sigma, sigma);
    __m256d drift = _mm256_fmadd_pd(half, sigma2, r);       // r + 0.5*sigma^2
    __m256d S_over_K = _mm256_div_pd(S, K);
    __m256d logSK = simdmath::log256(S_over_K);
    __m256d numer = _mm256_fmadd_pd(drift, T, logSK);       // log(S/K) + drift*T
    __m256d sigma_sqrtT = _mm256_mul_pd(sigma, sqrtT);
    __m256d d1 = _mm256_div_pd(numer, sigma_sqrtT);
    __m256d d2 = _mm256_sub_pd(d1, sigma_sqrtT);

    __m256d neg_rT = _mm256_mul_pd(_mm256_mul_pd(r, T), _mm256_set1_pd(-1.0));
    __m256d disc = simdmath::exp256(neg_rT);

    __m256d Nd1 = simdmath::norm_cdf256(d1);
    __m256d Nd2 = simdmath::norm_cdf256(d2);

    __m256d K_disc = _mm256_mul_pd(K, disc);
    __m256d call = _mm256_fmsub_pd(S, Nd1, _mm256_mul_pd(K_disc, Nd2));

    __m256d one_minus_Nd1 = _mm256_sub_pd(one, Nd1);
    __m256d one_minus_Nd2 = _mm256_sub_pd(one, Nd2);
    __m256d put = _mm256_fmsub_pd(K_disc, one_minus_Nd2, _mm256_mul_pd(S, one_minus_Nd1));

    _mm256_storeu_pd(call_out, call);
    _mm256_storeu_pd(put_out, put);
}

} // namespace bs
