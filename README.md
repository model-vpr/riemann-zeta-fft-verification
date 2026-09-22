[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22902045.svg)](https://doi.org/10.5281/zenodo.22902045)

# riemann-zeta-fft-verification
Python code for blind FFT detection of nontrivial zeros of the Riemann zeta function from prime distribution.

# Riemann Zeta FFT Verification

Python code for **blind FFT-based detection of nontrivial zeros of the Riemann zeta function** from the distribution of prime powers alone.

This repository accompanies the preprint:

> **Empirical Verification of the Riemann–von Mangoldt Explicit Formula via Fast Fourier Transform**
> Stefka Georgieva, VPR Research, 2026.
 DOI: [10.5281/zenodo.22902045](https://doi.org/10.5281/zenodo.22902045)

## Summary

Using only the Chebyshev function $\psi(x)$ — computed from a standard sieve of Eratosthenes — we:

1. Form the normalized deviation $(\psi(x) - x)/\sqrt{x}$.
2. Resample it in logarithmic space over $u = \ln x$.
3. Apply a Kaiser-windowed FFT.
4. Detect local maxima above a fixed amplitude threshold.
5. Compare the detected peaks with known zeros **only after detection** (blind test).

At $N = 5 \times 10^6$, the method resolves **39–40 of the top 40 nontrivial zeros** of $\zeta(s)$ within **1%**, without using any prior knowledge of their positions.

We also verify that the peak amplitudes follow the theoretical prediction

$$A(\gamma) \propto \frac{1}{|\rho|} = \frac{1}{\sqrt{1/4 + \gamma^2}},$$

with mean ratio **1.022** and standard deviation **0.155** across the first 80 zeros.

## Files

| File | Description |
|---|---|
| `blind_zeta_test.py` | Blind FFT detection of zeta zeros; no reference zeros used during detection. |
| `amplitude_vs_gamma.py` | Compares detected peak amplitudes with the theoretical $1/\sqrt{1/4 + \gamma^2}$ prediction. |


## Requirements

- Python 3.9+
- `numpy`
- `scipy`
- `matplotlib` (only for plotting)

Install:

```bash
pip install numpy scipy matplotlib
