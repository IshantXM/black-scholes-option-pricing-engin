// black_scholes_scalar.hpp
// Reference (ground-truth) scalar Black-Scholes pricer using std::exp,
// std::log, std::erf. This is the correctness baseline the AVX2 path is
// validated against, and the performance baseline it is benchmarked against.
#pragma once
#include <cmath>

namespace bs {

inline double norm_cdf(double x) {
    return 0.5 * std::erfc(-x * 0.7071067811865476);
}

// Returns (call_price, put_price)
inline void price_scalar(double S, double K, double r, double sigma, double T,
                          double& call, double& put) {
    double sqrtT = std::sqrt(T);
    double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
    double d2 = d1 - sigma * sqrtT;
    double disc = std::exp(-r * T);
    double Nd1 = norm_cdf(d1);
    double Nd2 = norm_cdf(d2);
    call = S * Nd1 - K * disc * Nd2;
    put  = K * disc * (1.0 - Nd2) - S * (1.0 - Nd1);
}

} // namespace bs
