#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <immintrin.h>
#include "../include/black_scholes_scalar.hpp"
#include "../include/black_scholes_avx2.hpp"

struct OptionSet {
    std::vector<double> S, K, r, sigma, T;
};

static OptionSet generate_options(size_t n, unsigned seed) {
    OptionSet o;
    o.S.resize(n); o.K.resize(n); o.r.resize(n); o.sigma.resize(n); o.T.resize(n);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> distS(20.0, 500.0);
    std::uniform_real_distribution<double> distMoneyness(0.6, 1.6); // K = S * moneyness
    std::uniform_real_distribution<double> distR(0.0, 0.10);
    std::uniform_real_distribution<double> distSigma(0.05, 1.20);
    std::uniform_real_distribution<double> distT(0.02, 3.0);
    for (size_t i = 0; i < n; ++i) {
        o.S[i] = distS(gen);
        o.K[i] = o.S[i] * distMoneyness(gen);
        o.r[i] = distR(gen);
        o.sigma[i] = distSigma(gen);
        o.T[i] = distT(gen);
    }
    return o;
}

int main(int argc, char** argv) {
    size_t N = 4'000'000; // must be multiple of 4 for the AVX2 loop below
    N -= (N % 4);

    printf("Generating %zu random option contracts...\n", N);
    OptionSet opt = generate_options(N, 42);

    std::vector<double> call_scalar(N), put_scalar(N);
    std::vector<double> call_avx2(N), put_avx2(N);

    // ---------------- Correctness check ----------------
    for (size_t i = 0; i < N; ++i) {
        bs::price_scalar(opt.S[i], opt.K[i], opt.r[i], opt.sigma[i], opt.T[i],
                          call_scalar[i], put_scalar[i]);
    }
    for (size_t i = 0; i < N; i += 4) {
        __m256d S = _mm256_loadu_pd(&opt.S[i]);
        __m256d K = _mm256_loadu_pd(&opt.K[i]);
        __m256d r = _mm256_loadu_pd(&opt.r[i]);
        __m256d sg = _mm256_loadu_pd(&opt.sigma[i]);
        __m256d T = _mm256_loadu_pd(&opt.T[i]);
        bs::price_avx2(S, K, r, sg, T, &call_avx2[i], &put_avx2[i]);
    }

    double sum_abs_err = 0.0, max_abs_err = 0.0;
    double sum_abs_err_put = 0.0, max_abs_err_put = 0.0;
    for (size_t i = 0; i < N; ++i) {
        double ec = std::fabs(call_scalar[i] - call_avx2[i]);
        double ep = std::fabs(put_scalar[i] - put_avx2[i]);
        sum_abs_err += ec; max_abs_err = std::max(max_abs_err, ec);
        sum_abs_err_put += ep; max_abs_err_put = std::max(max_abs_err_put, ep);
    }
    double mae_call = sum_abs_err / N;
    double mae_put = sum_abs_err_put / N;

    printf("\n=== Accuracy: AVX2 vs scalar (std::exp/log/erfc) reference ===\n");
    printf("Call:  MAE = %.6e   Max abs err = %.6e\n", mae_call, max_abs_err);
    printf("Put:   MAE = %.6e   Max abs err = %.6e\n", mae_put, max_abs_err_put);

    // ---------------- Benchmark: scalar ----------------
    const int reps = 5;
    double best_scalar_sec = 1e18, best_avx2_sec = 1e18;

    for (int rep = 0; rep < reps; ++rep) {
        auto t0 = std::chrono::high_resolution_clock::now();
        double sink = 0.0;
        for (size_t i = 0; i < N; ++i) {
            double c, p;
            bs::price_scalar(opt.S[i], opt.K[i], opt.r[i], opt.sigma[i], opt.T[i], c, p);
            sink += c + p;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();
        best_scalar_sec = std::min(best_scalar_sec, sec);
        if (sink == -1.0) printf(""); // prevent optimizing away
    }

    for (int rep = 0; rep < reps; ++rep) {
        auto t0 = std::chrono::high_resolution_clock::now();
        __m256d acc = _mm256_setzero_pd();
        for (size_t i = 0; i < N; i += 4) {
            __m256d S = _mm256_loadu_pd(&opt.S[i]);
            __m256d K = _mm256_loadu_pd(&opt.K[i]);
            __m256d r = _mm256_loadu_pd(&opt.r[i]);
            __m256d sg = _mm256_loadu_pd(&opt.sigma[i]);
            __m256d T = _mm256_loadu_pd(&opt.T[i]);
            double c4[4], p4[4];
            bs::price_avx2(S, K, r, sg, T, c4, p4);
            acc = _mm256_add_pd(acc, _mm256_loadu_pd(c4));
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();
        best_avx2_sec = std::min(best_avx2_sec, sec);
        alignas(32) double accbuf[4];
        _mm256_store_pd(accbuf, acc);
        if (accbuf[0] == -1.0) printf("");
    }

    double scalar_throughput = N / best_scalar_sec;
    double avx2_throughput = N / best_avx2_sec;

    printf("\n=== Performance (best of %d runs, N=%zu options) ===\n", reps, N);
    printf("Scalar : %.3f sec   (%.2f M options/sec)\n", best_scalar_sec, scalar_throughput / 1e6);
    printf("AVX2   : %.3f sec   (%.2f M options/sec)\n", best_avx2_sec, avx2_throughput / 1e6);
    printf("Speedup: %.2fx\n", best_scalar_sec / best_avx2_sec);

    return 0;
}
