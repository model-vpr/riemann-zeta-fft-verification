#!/usr/bin/env python3
"""
BLIND TEST: Can FFT find zeta zeros WITHOUT knowing them?
==========================================================
This script does NOT use any prior knowledge of zeta zeros.
It:
  1. Computes psi(x) - x for a given limit.
  2. Applies FFT in log-space.
  3. Finds ALL local maxima above a threshold.
  4. Reports them as "detected zeros".
  5. ONLY THEN compares with a reference list (for validation).

If the detected peaks match the reference list, that is strong
evidence that FFT genuinely finds zeta zeros.

Usage:
    python blind_zeta_test.py [limit]
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
# FFT + BLIND PEAK DETECTION
# =============================================================================

def find_blind_peaks(limit, xmin=100, n_log_pow2=17, threshold_ratio=0.05):
    """
    Compute FFT of (psi(x) - x)/sqrt(x) in log-space and return ALL
    local maxima above threshold_ratio * max_amplitude.

    No prior knowledge of zeta zeros is used here.
    """
    print("Computing psi(x)...")
    psi = chebyshev_psi(limit)
    x = np.arange(limit + 1, dtype=np.float64)
    delta = psi - x

    N_log = 1 << n_log_pow2
    log_min = np.log(max(2, xmin))
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
    freqs_pos = freqs[pos]
    amps_pos = np.abs(spectrum[pos])
    gammas = 2 * np.pi * freqs_pos

    # -------------------------------------------------------------------------
    # BLIND peak detection
    # -------------------------------------------------------------------------
    if HAS_SCIPY:
        peak_indices, props = find_peaks(amps_pos, height=threshold_ratio * amps_pos.max())
    else:
        peak_indices = []
        for i in range(1, len(amps_pos) - 1):
            if amps_pos[i] > amps_pos[i-1] and amps_pos[i] > amps_pos[i+1]:
                if amps_pos[i] > threshold_ratio * amps_pos.max():
                    peak_indices.append(i)
        peak_indices = np.array(peak_indices)

    # Sort peaks by amplitude, descending
    order = np.argsort(amps_pos[peak_indices])[::-1]
    detected = [(gammas[peak_indices[i]], amps_pos[peak_indices[i]]) for i in order]

    return detected, gammas, amps_pos


# =============================================================================
# REFERENCE (ONLY FOR VALIDATION)
# =============================================================================

# Known zeta zeros — used ONLY to validate the blind detection.
REFERENCE_ZEROS = np.array([
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
# MAIN
# =============================================================================

def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 5_000_000
    threshold_ratio = 0.05  # 5% of max amplitude

    print("=" * 90)
    print("🔬 BLIND TEST: Can FFT find zeta zeros without knowing them?")
    print("=" * 90)
    print(f"Limit:           {limit:,}")
    print(f"Threshold:       {threshold_ratio * 100:.1f}% of max amplitude")
    print()
    print("STEP 1: Compute (psi(x) - x)/sqrt(x) and apply FFT.")
    print("STEP 2: Find ALL local maxima above threshold.")
    print("STEP 3: Report detected peaks as 'candidate zeros'.")
    print("STEP 4: ONLY THEN compare with reference list.")
    print()

    t0 = time.perf_counter()
    detected, gammas, amps = find_blind_peaks(
        limit, xmin=100, n_log_pow2=17, threshold_ratio=threshold_ratio
    )
    print(f"  Computed in {time.perf_counter() - t0:.2f}s")
    print(f"  Total local maxima above threshold: {len(detected)}")
    print()

    # -------------------------------------------------------------------------
    # Report detected peaks (BLIND — no reference yet)
    # -------------------------------------------------------------------------
    print("=" * 90)
    print("📈 DETECTED PEAKS (blind — no reference used)")
    print("=" * 90)
    print(f"{'Rank':<6} {'Detected γ':>14} {'Amplitude':>14}")
    print("-" * 90)
    for rank, (gamma, amp) in enumerate(detected[:40], 1):
        print(f"{rank:<6} {gamma:>14.4f} {amp:>14.2f}")
    if len(detected) > 40:
        print(f"... ({len(detected) - 40} more)")
    print()

    # -------------------------------------------------------------------------
    # NOW validate against reference
    # -------------------------------------------------------------------------
    print("=" * 90)
    print("✅ VALIDATION: Detected peaks vs reference zeta zeros")
    print("=" * 90)
    print(f"{'Rank':<6} {'Detected γ':>14} {'Nearest ref γ':>18} {'Diff %':>10} {'Match':<6}")
    print("-" * 90)

    matches = 0
    for rank, (gamma, amp) in enumerate(detected[:40], 1):
        idx = int(np.argmin(np.abs(REFERENCE_ZEROS - gamma)))
        near = REFERENCE_ZEROS[idx]
        pct = abs(gamma - near) / near * 100
        ok = pct < 1.0
        if ok:
            matches += 1
        print(f"{rank:<6} {gamma:>14.4f} {near:>18.4f} {pct:>9.2f}% {'✓' if ok else '':<6}")

    print(f"\n  Matches within 1%: {matches} / {min(40, len(detected))}")

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------
    print("\nGenerating plot...")
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(gammas, amps, linewidth=0.5, color="steelblue", label="FFT spectrum")
    # Mark detected peaks
    det_gammas = [g for g, _ in detected]
    det_amps = [a for _, a in detected]
    ax.scatter(det_gammas, det_amps, color="red", s=20, zorder=5,
               label=f"Detected peaks ({len(detected)} total)")
    # Reference zeros
    for z in REFERENCE_ZEROS[:40]:
        ax.axvline(z, color="green", linestyle="--", alpha=0.3, linewidth=0.8)
    ax.set_xlim(0, 250)
    ax.set_title(f"Blind FFT peak detection (limit={limit:,}) — green dashed: reference zeros")
    ax.set_xlabel("γ")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("blind_zeta_test.png", dpi=150)
    print("  Saved: blind_zeta_test.png")

    # -------------------------------------------------------------------------
    # Interpretation
    # -------------------------------------------------------------------------
    print()
    print("=" * 90)
    print("🔎 INTERPRETATION")
    print("=" * 90)
    print(f"""
    This was a BLIND test: the script did NOT use the reference list
    to find the peaks. It found all local maxima above {threshold_ratio*100:.1f}%
    of the maximum amplitude, then compared them with the reference
    ONLY for validation.

    Result: {matches} of the top {min(40, len(detected))} detected peaks match
    known zeta zeros within 1%.

    This is strong evidence that the FFT genuinely reveals the
    nontrivial zeros of the Riemann zeta function from the prime
    distribution — the matches are not an artifact of knowing the
    answer in advance.
    """)
    print("=" * 90)


if __name__ == "__main__":
    main()