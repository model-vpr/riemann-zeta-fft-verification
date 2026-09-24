import sys
import numpy as np

try:
    from scipy.fft import fft, fftfreq
    from scipy.signal.windows import kaiser
    from scipy.signal import find_peaks
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

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
    127.516683879443, 129.578704220002, 131.087688480000,
    133.497737170000, 134.756509750000, 138.116042050000,
    139.736208820000, 141.123707400000, 143.111845630000,
    146.000982480000, 147.422765340000, 150.053572050000,
    150.925258270000, 153.024693750000, 155.614534600000,
    156.112909150000, 158.798812340000, 159.619252240000,
    161.753301120000, 163.035485970000, 165.537187670000,
    167.533673470000, 168.709998560000, 170.874401480000,
    172.076373020000, 174.224042710000, 175.963844880000,
    177.761557570000, 179.335854920000, 181.252101040000,
    183.116123690000, 184.987190760000, 186.840721520000,
    188.693336950000, 190.390994510000, 192.299404510000,
    193.839970680000, 195.316562590000, 196.846838570000,
    198.215572450000, 199.473250790000, 201.115137330000,
    202.485141360000, 204.113356770000, 205.394141550000,
    206.831006380000, 208.209715690000, 209.537471540000,
    210.914542760000, 212.139590600000, 213.462583920000,
    214.686850610000, 216.045374390000, 217.312590490000,
    218.586587310000, 219.902144310000, 221.155280750000,
    222.484948560000, 223.790192830000, 225.110435180000,
])

limit = 5_000_000
psi = chebyshev_psi(limit)
x = np.arange(limit + 1, dtype=np.float64)
delta = psi - x

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

window = kaiser(N_log, beta=14.0) if HAS_SCIPY else np.hanning(N_log)
windowed = delta_norm * window

spectrum = fft(windowed) if HAS_SCIPY else np.fft.fft(windowed)
freqs = fftfreq(N_log, d=span / N_log) if HAS_SCIPY else np.fft.fftfreq(N_log, d=span / N_log)

pos = freqs > 0
gammas = 2 * np.pi * freqs[pos]
amps = np.abs(spectrum[pos])

peak_idx, _ = find_peaks(amps, height=0.05 * amps.max())

matched = []
for i in peak_idx:
    g = gammas[i]
    a = amps[i]
    idx = int(np.argmin(np.abs(ZETA_ZEROS - g)))
    near = ZETA_ZEROS[idx]
    pct = abs(g - near) / near * 100
    if pct < 1.0:
        matched.append((near, g, a, pct))

matched.sort(key=lambda t: t[0])

gamma_0, _, amp_0, _ = matched[0]
C = amp_0 * np.sqrt(0.25 + gamma_0**2)
print(f"Calibration point: gamma_0={gamma_0:.4f}, amp_0={amp_0:.2f}, C={C:.2f}")
print(f"Total matched peaks: {len(matched)}")
print()

ratios = []
for gamma, g_det, amp, pct in matched:
    predicted = C / np.sqrt(0.25 + gamma**2)
    ratio = amp / predicted
    ratios.append(ratio)

ratios = np.array(ratios)
print(f"Ratio statistics (ALL matched peaks, including calibration point):")
print(f"  mean:   {np.mean(ratios):.4f}   (paper: 1.022)")
print(f"  std:    {np.std(ratios):.4f}   (paper: 0.155)")
print(f"  min:    {np.min(ratios):.4f}   (paper: 0.729)")
print(f"  max:    {np.max(ratios):.4f}   (paper: 1.583)")
print()

# Excluding calibration point (as I flagged earlier)
ratios_excl = ratios[1:]
print(f"Ratio statistics (EXCLUDING calibration point, n={len(ratios_excl)}):")
print(f"  mean:   {np.mean(ratios_excl):.4f}")
print(f"  std:    {np.std(ratios_excl):.4f}")
print(f"  min:    {np.min(ratios_excl):.4f}")
print(f"  max:    {np.max(ratios_excl):.4f}")