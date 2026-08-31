import numpy as np
import math

def fit(f, a, b, deg):
    xs_fit = np.cos(np.pi*(np.arange(2*deg+1)+0.5)/(2*deg+1))
    xs_fit_mapped = 0.5*(b-a)*xs_fit + 0.5*(b+a)
    ys_fit = np.array([f(x) for x in xs_fit_mapped], dtype=np.float64)
    cheb = np.polynomial.chebyshev.Chebyshev.fit(xs_fit_mapped, ys_fit, deg, domain=[a,b])
    poly = cheb.convert(kind=np.polynomial.Polynomial)
    return poly.coef

lo_m, hi_m = math.sqrt(0.5), math.sqrt(2.0)
u_lo, u_hi = lo_m - 1.0, hi_m - 1.0
c_log = fit(math.log1p, u_lo, u_hi, 16)
print("LOG1P deg16 coeffs (c0..c16):")
for i,c in enumerate(c_log):
    print(f"c{i} = {c!r}")

def g(x):
    if x == 0: return 1.0
    return math.erfc(x)*math.exp(x*x)
c_erf = fit(g, 0.0, 6.0, 18)
print()
print("ERF-G deg18 coeffs (c0..c18):")
for i,c in enumerate(c_erf):
    print(f"c{i} = {c!r}")

# verify
xs = np.linspace(u_lo, u_hi, 200000)
true = np.array([math.log1p(x) for x in xs])
approx = np.polyval(c_log[::-1], xs)
print("\nlog1p max abs err:", np.max(np.abs(true-approx)))

xs2 = np.linspace(0,6,500000)
true_erf = np.array([math.erf(x) for x in xs2])
approx_erf = 1.0 - np.polyval(c_erf[::-1], xs2)*np.exp(-xs2**2)
print("erf max abs err:", np.max(np.abs(true_erf-approx_erf)))
