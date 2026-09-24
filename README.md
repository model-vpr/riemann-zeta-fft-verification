[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22902045.svg)](https://doi.org/10.5281/zenodo.22932353)

# Riemann Zeta FFT Verification

Python code for **blind FFT-based detection of nontrivial zeros of the Riemann zeta function** from the distribution of prime powers alone.

This repository accompanies the preprint:

> **Empirical Verification of the Riemann–von Mangoldt Explicit Formula via Fast Fourier Transform**
> Stefka Georgieva, VPR Research, 2026.
> DOI: [10.5281/zenodo.22932353](https://doi.org/10.5281/zenodo.22932353)

## Summary

Using only the Chebyshev function $\psi(x)$ — computed from a standard sieve of Eratosthenes — we:

1. Form the normalized deviation $(\psi(x) - x)/\sqrt{x}$.
2. Resample it in logarithmic space over $u = \ln x$.
3. Apply a Kaiser-windowed FFT.
4. Detect local maxima above a fixed amplitude threshold.
5. Compare the detected peaks with known zeros **only after detection** (blind test).

At $N = 5 \times 10^6$, the method resolves **39–40 of the top 40 nontrivial zeros** of $\zeta(s)$ within **1%**, without using any prior knowledge of their positions. The result is stable across a sixfold range of the log-window lower bound $x_{\min} \in \{50, 100, 200, 300\}$.

We also verify that the peak amplitudes follow the theoretical prediction

$$A(\gamma) \propto \frac{1}{|\rho|} = \frac{1}{\sqrt{1/4 + \gamma^2}},$$

with mean ratio **1.023** and standard deviation **0.156** across **77 matched zeros** (excluding the calibration point — the strongest peak, whose ratio is 1.000 by construction rather than by measurement). Including the calibration point (n = 78) gives mean **1.022** and standard deviation **0.155**; the difference is in the fourth decimal place and does not affect any conclusion. See the preprint, Section 3.2, for details.

## Files

| File | Description |
|---|---|
| `blind_zeta_test.py` | Blind FFT detection of zeta zeros; no reference zeros used during detection. |
| `amplitude_vs_gamma.py` | Compares detected peak amplitudes with the theoretical $C/\sqrt{1/4 + \gamma^2}$ prediction.
| `figures/blind_zeta_test.png` | Output plot: detected peaks vs. reference zeros. |
| `figures/amplitude_vs_gamma.png` | Output plot: amplitude scaling analysis. |

## Requirements

- Python 3.9+
- `numpy`
- `scipy`
- `matplotlib` (only for plotting)

Install:

```bash
pip install numpy scipy matplotlib
```

## Usage

```bash
python blind_zeta_test.py
python amplitude_vs_gamma.py
```

Both scripts run in under one second and use approximately 40 MB of memory at the default settings ($N = 5 \times 10^6$).

## Citation

If you use this code, please cite the accompanying preprint:

```bibtex
@misc{georgieva2026riemann,
  author       = {Georgieva, Stefka},
  title        = {Empirical Verification of the Riemann--von Mangoldt Explicit Formula via Fast Fourier Transform},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.22932353},
  url          = {https://doi.org/10.5281/zenodo.22932353}
}
```

## License

MIT
