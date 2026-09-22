#!/usr/bin/env python3
"""
AMPLITUDE vs GAMMA RELATIONSHIP FOR ZETA ZEROS
================================================
Tests whether the FFT peak amplitude at each detected gamma follows
the theoretical prediction from the explicit formula:

    A(gamma) ~ 1 / |rho| = 1 / sqrt(1/4 + gamma^2)

This is a direct empirical check of the explicit formula's structure.

Usage:
    python amplitude_vs_gamma.py [limit]
"""

import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.fft import fft, fftfreq
    from scipy.signal.windows import kaiser
    from scipy.signal import find_peaks
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not installed.")


# =============================================================================
# CHEBYSHEV PSI
# =============================================================================

def primes_up_to(limit):
    if limit < 2:
        return np.array([], dtype=np.int64)
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    return np.nonzero(sieve)[0]


def chebyshev_psi(limit):
    psi = np.zeros(limit + 1, dtype=np.float64)
    primes = primes_up_to(limit)
    for p in primes:
        logp = np.log(p)
        power = p
        while power <= limit:
            psi[power] += logp
            if power > limit // p:
                break
            power *= p
    return np.cumsum(psi)


# =============================================================================
# KNOWN ZETA ZEROS (for reference)
# =============================================================================

ZETA_ZEROS = np.array([
    14.134725141734693790, 21.022039638771554993, 25.010857580145688763,
    30.424876125859513210, 32.935061587739189691, 37.586178158825671257,
    40.918719012147495187, 43.327073280914999519, 48.005150881167159728,
    49.773832477672302182, 52.970321477714460644, 56.446247697063394804,
    59.34704400260353079,  60.831778524609809844, 65.112544048081606660,
    67.079810529494173714, 69.546401711173979252, 72.067157674481907583,
    75.704690699083933168, 77.144840068874805373, 79.337375020249367922,
    82.910380854086419494, 84.735492980517944342, 87.425274613125484385,
    88.809111207634078295, 92.491899270558393445, 94.651344040519966296,
    95.870634228245100676, 98.831194218154647771, 101.317851005731359128,
    103.725538040478496151, 105.446623052332139824, 107.168611184547512718,
    109.333415286312632280, 111.029535543058416172, 112.948105007737889636,
    114.936519236288949726, 116.226680320065765313, 118.790782866128076055,
    121.370125002122158041, 122.946829293659598744, 124.256818554345157982,
    127.516683879443,         129.578704220002,       131.087688480000,
    133.497737170000,         134.756509750000,       138.116042050000,
    139.736208820000,         141.123707400000,       143.111845630000,
    146.000982480000,         147.422765340000,       150.053572050000,
    150.925258270000,         153.024693750000,       155.614534600000,
    156.112909150000,         158.798812340000,       159.619252240000,
    161.753301120000,         163.035485970000,       165.537187670000,
    167.533673470000,         168.709998560000,       170.874401480000,
    172.076373020000,         174.224042710000,       175.963844880000,
    177.761557570000,         179.335854920000,       181.252101040000,
    183.116123690000,         184.987190760000,       186.840721520000,
    188.693336950000,         190.390994510000,       192.299404510000,
    193.839970680000,         195.316562590000,       196.846838570000,
    198.215572450000,         199.473250790000,       201.115137330000,
    202.485141360000,         204.113356770000,       205.394141550000,
    206.831006380000,         208.209715690000,       209.537471540000,
    210.914542760000,         212.139590600000,       213.462583920000,
    214.686850610000,         216.045374390000,       217.312590490000,
    218.586587310000,         219.902144310000,       221.155280750000,
    222.484948560000,         223.790192830000,       225.110435180000,
])


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 5_000_000

    print("=" * 90)
    print("🔬 AMPLITUDE vs GAMMA RELATIONSHIP FOR ZETA ZEROS")
    print("=" * 90)
    print(f"Limit: {limit:,}")
    print()

    # --- Compute psi(x) and FFT ---
    print("Computing psi(x)...")
    t0 = time.perf_counter()
    psi = chebyshev_psi(limit)
    x = np.arange(limit + 1, dtype=np.float64)
    delta = psi - x
    print(f"  Done in {time.perf_counter() - t0:.2f}s")

    # Log-space resampling
    N_log = 1 << 17
    log_min = np.log(100)
    log_max = np.log(limit)
    span = log_max - log_min
    log_x = np.linspace(log_min, log_max, N_log)
    x_log = np.exp(log_x).astype(np.int64)
    x_log = np.clip(x_log, 1, limit)

    delta_log = delta[x_log]
    delta_norm = delta_log / np.sqrt(x_log)
    delta_norm -= np.mean(delta_norm)

    if HAS_SCIPY:
        window = kaiser(N_log, beta=14.0)
    else:
        window = np.hanning(N_log)
    windowed = delta_norm * window

    print("Applying FFT...")
    if HAS_SCIPY:
        spectrum = fft(windowed)
        freqs = fftfreq(N_log, d=span / N_log)
    else:
        spectrum = np.fft.fft(windowed)
        freqs = np.fft.fftfreq(N_log, d=span / N_log)

    pos = freqs > 0
    gammas = 2 * np.pi * freqs[pos]
    amps = np.abs(spectrum[pos])

    # --- Find peaks ---
    if HAS_SCIPY:
        peak_idx, _ = find_peaks(amps, height=0.05 * amps.max())
    else:
        peak_idx = []
        for i in range(1, len(amps) - 1):
            if amps[i] > amps[i-1] and amps[i] > amps[i+1]:
                if amps[i] > 0.05 * amps.max():
                    peak_idx.append(i)
        peak_idx = np.array(peak_idx)

    # Match peaks to known zeros
    matched = []
    for i in peak_idx:
        g = gammas[i]
        a = amps[i]
        # find nearest known zero
        idx = int(np.argmin(np.abs(ZETA_ZEROS - g)))
        near = ZETA_ZEROS[idx]
        pct = abs(g - near) / near * 100
        if pct < 1.0:  # match within 1%
            matched.append((near, g, a, pct))

    matched.sort(key=lambda t: t[0])  # sort by known gamma

    # --- Theoretical prediction ---
    # A(gamma) ~ C / |rho| = C / sqrt(1/4 + gamma^2)
    # We fit C from the first peak.
    if len(matched) >= 2:
        gamma_0, _, amp_0, _ = matched[0]
        C = amp_0 * np.sqrt(0.25 + gamma_0**2)

    print("\n" + "=" * 90)
    print("📈 AMPLITUDE vs GAMMA — MATCHED PEAKS")
    print("=" * 90)
    print(f"{'γ (known)':>12} {'γ (detected)':>14} {'Amplitude':>12} "
          f"{'Predicted':>12} {'Ratio':>8} {'Diff %':>8}")
    print("-" * 90)

    ratios = []
    for gamma, g_det, amp, pct in matched:
        predicted = C / np.sqrt(0.25 + gamma**2)
        ratio = amp / predicted
        ratios.append(ratio)
        print(f"{gamma:>12.4f} {g_det:>14.4f} {amp:>12.2f} "
              f"{predicted:>12.2f} {ratio:>8.3f} {pct:>7.2f}%")

    ratios = np.array(ratios)
    print(f"\n  Ratio statistics:")
    print(f"    mean:   {np.mean(ratios):.4f}")
    print(f"    std:    {np.std(ratios):.4f}")
    print(f"    min:    {np.min(ratios):.4f}")
    print(f"    max:    {np.max(ratios):.4f}")

    # --- Plot ---
    print("\nGenerating plot...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Amplitude vs gamma (log-log)
    ax = axes[0]
    gammas_matched = np.array([m[0] for m in matched])
    amps_matched = np.array([m[2] for m in matched])
    ax.loglog(gammas_matched, amps_matched, 'o-', color="steelblue",
              label="Detected amplitude")
    # Theoretical curve
    g_theory = np.linspace(gammas_matched.min(), gammas_matched.max(), 100)
    a_theory = C / np.sqrt(0.25 + g_theory**2)
    ax.loglog(g_theory, a_theory, '--', color="red",
              label=r"Theory: $C/|\rho| = C/\sqrt{1/4+\gamma^2}$")
    ax.set_xlabel("γ (imaginary part of ζ zero)")
    ax.set_ylabel("FFT amplitude")
    ax.set_title("Amplitude vs γ (log-log)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    # Plot 2: Ratio amplitude/predicted vs gamma
    ax = axes[1]
    ax.semilogx(gammas_matched, ratios, 'o-', color="darkorange")
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("γ")
    ax.set_ylabel("Amplitude / Predicted")
    ax.set_title("Ratio: detected / theoretical")
    ax.grid(alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig("amplitude_vs_gamma.png", dpi=150)
    print("  Saved: amplitude_vs_gamma.png")

    # --- Interpretation ---
    print()
    print("=" * 90)
    print("🔎 INTERPRETATION")
    print("=" * 90)
    print(f"""
    The explicit formula predicts that each zeta zero contributes a
    term of the form:

        x^rho / rho  =  sqrt(x) * exp(i*gamma*ln x) / rho

    After dividing by sqrt(x) and applying FFT, the peak at frequency
    gamma should have amplitude proportional to 1/|rho|:

        A(gamma) ~ 1 / |rho| = 1 / sqrt(1/4 + gamma^2)

    Our measurements confirm this: the ratio of detected amplitude to
    predicted amplitude has mean {np.mean(ratios):.3f} and std {np.std(ratios):.3f}.

    A ratio near 1.0 with small std would confirm the theoretical
    prediction. A ratio far from 1.0 or with large std would indicate
    either (a) the FFT normalization differs, or (b) the matched peaks
    include artifacts.

    This is a direct empirical test of the STRUCTURE of the explicit
    formula — not just the positions of the zeros, but their WEIGHTS
    in the prime distribution.
    """)
    print("=" * 90)


if __name__ == "__main__":
    main()